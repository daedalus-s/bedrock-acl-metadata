import json
import os
import logging
import boto3
from pinecone import Pinecone, ServerlessSpec
from typing import Dict, List, Any
from urllib.parse import unquote_plus

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize clients
pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
bedrock = boto3.client('bedrock-runtime')
s3_client = boto3.client('s3')

# Constants
INDEX_NAME = os.environ['PINECONE_INDEX_NAME']
DIMENSION = 1536  # Titan embedding dimension

def clean_text(text: str) -> str:
    """
    Cleans and normalizes input text.
    """
    return ''.join(char for char in text if char.isprintable() or char.isspace())

def ensure_index_exists():
    """
    Gets the Pinecone index instance.
    """
    try:
        return pinecone_client.Index(INDEX_NAME)
    except Exception as e:
        logging.error(f"Error accessing index: {str(e)}")
        raise

def get_s3_object_acl(bucket: str, key: str) -> Dict[str, str]:
    """
    Get ACL information for an S3 object and return flattened structure.
    """
    try:
        acl = s3_client.get_object_acl(Bucket=bucket, Key=key)
        return {
            'owner_id': acl['Owner']['ID'],
            'permissions': ','.join([
                f"{grant['Grantee'].get('ID', grant['Grantee'].get('URI'))}:{grant['Permission']}" 
                for grant in acl['Grants']
            ])
        }
    except Exception as e:
        logging.warning(f"Could not get ACL for {bucket}/{key}: {str(e)}")
        return {
            'owner_id': 'unknown',
            'permissions': ''
        }

def get_s3_object_tags(bucket: str, key: str) -> Dict[str, str]:
    """
    Get tags for an S3 object.
    """
    try:
        tags = s3_client.get_object_tagging(Bucket=bucket, Key=key)
        return {tag['Key']: tag['Value'] for tag in tags['TagSet']}
    except Exception as e:
        logging.warning(f"Could not get tags for {bucket}/{key}: {str(e)}")
        return {}

def get_embedding(text: str) -> List[float]:
    """
    Get embeddings from Amazon Bedrock Titan Embeddings model.
    """
    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "inputText": text
            })
        )
        embedding = json.loads(response['body'].read())['embedding']
        return embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        raise

def process_s3_object(bucket: str, key: str) -> Dict[str, Any]:
    """
    Process an S3 object with improved error handling and metadata flattening.
    """
    try:
        # Log attempt to access file
        logging.info(f"Attempting to access s3://{bucket}/{key}")
        
        # Clean up the key (remove any double slashes or path issues)
        key = key.lstrip('/').replace('//', '/')
        
        try:
            # First try to check if object exists
            s3_client.head_object(Bucket=bucket, Key=key)
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.error(f"Object not found: s3://{bucket}/{key}")
                return None
            elif e.response['Error']['Code'] == '403':
                logging.error(f"Permission denied: s3://{bucket}/{key}")
                return None
            else:
                raise

        # Get object content
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = clean_text(response['Body'].read().decode('utf-8'))
        
        # Get ACLs and tags
        acl_info = get_s3_object_acl(bucket, key)
        tags = get_s3_object_tags(bucket, key)
        
        # Generate embedding
        embedding = get_embedding(content)
        
        # Prepare metadata with flattened structure
        metadata = {
            "source": f"s3://{bucket}/{key}",
            "content_type": response.get('ContentType', 'text/plain'),
            "last_modified": response['LastModified'].isoformat(),
            "size": str(response['ContentLength']),
            "owner_id": acl_info['owner_id'],
            "permissions": acl_info['permissions'],
            "content": content  # Store full content for retrieval
        }
        
        # Add tags as top-level metadata with prefix
        for tag_key, tag_value in tags.items():
            metadata[f"tag_{tag_key}"] = tag_value
        
        return {
            "id": f"{bucket}/{key}",
            "vector": embedding,
            "metadata": metadata
        }
    except Exception as e:
        logging.error(f"Error processing {bucket}/{key}: {str(e)}", exc_info=True)
        return None

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler to process S3 events and update Pinecone index.
    """
    try:
        logging.info(f"Processing event: {json.dumps(event)}")
        index = ensure_index_exists()
        
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])
            
            if record['eventName'].startswith('ObjectCreated'):
                document_data = process_s3_object(bucket, key)
                if document_data:
                    # Upsert to Pinecone
                    index.upsert(
                        vectors=[(
                            document_data['id'],
                            document_data['vector'],
                            document_data['metadata']
                        )]
                    )
                    logging.info(f"Successfully processed and uploaded {bucket}/{key}")
                    
            elif record['eventName'].startswith('ObjectRemoved'):
                index.delete(ids=[f"{bucket}/{key}"])
                logging.info(f"Successfully deleted {bucket}/{key} from index")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Successfully processed S3 event')
        }
        
    except Exception as e:
        logging.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing S3 event: {str(e)}')
        }