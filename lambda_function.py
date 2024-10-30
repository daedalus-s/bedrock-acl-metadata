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
MAX_TOKENS = 300  # Maximum tokens per chunk
OVERLAP_PERCENTAGE = 20  # Percentage of overlap between chunks

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

def split_into_chunks(text: str, max_tokens: int = MAX_TOKENS, 
                     overlap_percentage: int = OVERLAP_PERCENTAGE) -> List[str]:
    """
    Split text into overlapping chunks.
    Args:
        text: Input text to split
        max_tokens: Maximum number of tokens per chunk (approximate by words)
        overlap_percentage: Percentage of overlap between chunks
    Returns:
        List of text chunks
    """
    words = text.split()
    chunk_size = max_tokens
    overlap_size = int(chunk_size * overlap_percentage / 100)
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        start = start + chunk_size - overlap_size
    
    logging.info(f"Split text into {len(chunks)} chunks with {overlap_percentage}% overlap")
    return chunks

def process_s3_object(bucket: str, key: str) -> List[Dict[str, Any]]:
    """
    Process an S3 object with chunking and improved error handling.
    """
    try:
        logging.info(f"Attempting to access s3://{bucket}/{key}")
        
        # Clean up the key
        key = key.lstrip('/').replace('//', '/')
        
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except s3_client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['404', '403']:
                logging.error(f"Access error ({error_code}) for s3://{bucket}/{key}")
                return None
            raise

        # Get object content
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = clean_text(response['Body'].read().decode('utf-8'))
        
        # Get ACLs and tags
        acl_info = get_s3_object_acl(bucket, key)
        tags = get_s3_object_tags(bucket, key)
        
        # Split content into chunks
        chunks = split_into_chunks(content)
        chunk_data = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Generate embedding for chunk
            embedding = get_embedding(chunk)
            
            # Prepare metadata
            metadata = {
                "source": f"s3://{bucket}/{key}",
                "content_type": response.get('ContentType', 'text/plain'),
                "last_modified": response['LastModified'].isoformat(),
                "size": str(response['ContentLength']),
                "owner_id": acl_info['owner_id'],
                "permissions": acl_info['permissions'],
                "content": chunk,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks))
            }
            
            # Add tags
            for tag_key, tag_value in tags.items():
                metadata[f"tag_{tag_key}"] = tag_value
            
            chunk_data.append({
                "id": f"{bucket}/{key}/chunk_{i}",
                "vector": embedding,
                "metadata": metadata
            })
            
            logging.info(f"Processed chunk {i+1}/{len(chunks)} for {bucket}/{key}")
        
        return chunk_data
    
    except Exception as e:
        logging.error(f"Error processing {bucket}/{key}: {str(e)}", exc_info=True)
        return None

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler to process S3 events and update Pinecone index with chunked data.
    """
    try:
        logging.info(f"Processing event: {json.dumps(event)}")
        index = ensure_index_exists()
        
        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = unquote_plus(record['s3']['object']['key'])
            
            if record['eventName'].startswith('ObjectCreated'):
                chunks_data = process_s3_object(bucket, key)
                if chunks_data:
                    # Prepare vectors for batch upsert
                    vectors_to_upsert = [
                        (chunk['id'], chunk['vector'], chunk['metadata'])
                        for chunk in chunks_data
                    ]
                    
                    # Upsert in batches of 100
                    batch_size = 100
                    for i in range(0, len(vectors_to_upsert), batch_size):
                        batch = vectors_to_upsert[i:i + batch_size]
                        index.upsert(vectors=batch)
                        logging.info(f"Upserted batch {i//batch_size + 1} for {bucket}/{key}")
                    
                    logging.info(f"Successfully processed {len(chunks_data)} chunks for {bucket}/{key}")
                    
            elif record['eventName'].startswith('ObjectRemoved'):
                # Delete all chunks for this document
                index.delete(filter={
                    "source": f"s3://{bucket}/{key}"
                })
                logging.info(f"Successfully deleted all chunks for {bucket}/{key}")
        
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