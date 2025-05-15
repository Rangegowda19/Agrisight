import json
import boto3

# Initialize DynamoDB client
dynamodb = boto3.client('dynamodb')
table_name = "speciesdeatils"  # Update with actual table name

def lambda_handler(event, context):
    print(f"Received event: {event}")  # Log event for debugging

    # Handle CORS preflight request
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps({'message': 'CORS preflight response'})
        }

    # Access query parameters
    query_params = event.get('queryStringParameters', {})
    class_name = query_params.get('class') or query_params.get('Class')  # Handle both cases

    if not class_name:
        return {
            'statusCode': 400,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps({'message': 'Query parameter "class" is required'})
        }

    try:
        print(f"Querying DynamoDB for class: {class_name}")  # Log class_name for debugging
        response = dynamodb.get_item(
            TableName=table_name,
            Key={
                'Class': {'S': class_name}
            }
        )

        if 'Item' in response:
            item = response['Item']
            print(f"Item found: {item}")  # Log retrieved item

            parsed_item = {k: parse_dynamodb_value(v) for k, v in item.items()}
            filtered_details = {k: v for k, v in parsed_item.items() if v and v != class_name}

            print(f"Returning response: {filtered_details}")  # Log the final response

            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'GET,OPTIONS'
                },
                'body': json.dumps(filtered_details)
            }
        else:
            return {
                'statusCode': 404,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'GET,OPTIONS'
                },
                'body': json.dumps({'message': 'Data not found for the given class'})
            }

    except Exception as e:
        print(f"Error querying DynamoDB: {e}")  # Log error details
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps({'message': 'Internal server error', 'details': str(e)})
        }

def parse_dynamodb_value(value):
    """ Converts a DynamoDB item value to a normal Python value. """
    if 'S' in value:
        return value['S']
    elif 'N' in value:
        return float(value['N']) if '.' in value['N'] else int(value['N'])
    elif 'BOOL' in value:
        return value['BOOL']
    elif 'M' in value:
        return {k: parse_dynamodb_value(v) for k, v in value['M'].items()}
    elif 'L' in value:
        return [parse_dynamodb_value(v) for v in value['L']]
    elif 'NULL' in value:
        return None
    else:
        return str(value)  # fallback for unknown types
