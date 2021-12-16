# -*- coding: utf-8 -*-

from google.cloud import pubsub_v1
import os


from google.cloud import pubsub_v1
import os

from src.utils.env import get_env_var
from src.utils.constants import TOPIC_ID


def publish_to_topic(msg):
    project_id = get_env_var('GCP_PROJECT')
    topic_id = TOPIC_ID
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    data = msg.encode("utf-8")
    future = publisher.publish(topic_path, data)
    print(future.result())
    print(f"Published messages to {topic_path}.")
    return future
