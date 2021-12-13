import base64
import json
from google.cloud import bigquery

def initial_method(event, context):
    # Decode pusub message
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    score_message = json.loads(pubsub_message)
    print("Received message: %s"%(str(score_message)))

        # Construct a BigQuery client object.
    client = bigquery.Client()
    # TODO(developer): Set table_id to the ID of table to append to.
    table_id = "mlops-uea-2.credito.scores"
    rows_to_insert = [score_message]
    errors = client.insert_rows_json(table_id, rows_to_insert)  # Make an API request.
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))

