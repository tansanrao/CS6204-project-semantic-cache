from __future__ import annotations
import logging

from policy.features import FeatureExtractor
from policy.ner import build_default_entity_recognizer
from policy.classifier import load_or_initialize_linucb
from policy.classifier import PolicyRuntimeConfig

from helper import get_train_test, llm_with_tool_call

from pathlib import Path

LOG = logging.getLogger("main")
logging._Level = 'DEBUG'
DEBUG = False

snapshot_path = Path('data/model_snapshots/linucb_snapshot.json')
feature_extractor = FeatureExtractor(entity_recognizer=build_default_entity_recognizer())
ttl_buckets = (5, 1440, 10080, 43200)  
    # in minutes: 5 mins, 1 day, 7 days, 30 days       
runtime_config = PolicyRuntimeConfig(
    policy_type="linucb",
    num_actions=len(ttl_buckets),
    feature_dimension=256,
    allow_feature_growth=True,
    linucb_alpha=0.6,
    linucb_regularization=1.0,
    linucb_min_propensity=1e-3,
    ts_regularization=1.0,
    ts_sampling_variance=1.0,
    ts_min_propensity=1e-3,
    snapshot_path=snapshot_path,
    autosave_interval=1)
policy = load_or_initialize_linucb(runtime_config, snapshot_path)

train_df, test_df = get_train_test('data/data.csv')
train_data = train_df.to_dict(orient="records")
test_data = test_df.to_dict(orient="records")

print(f"Train: {len(train_data)}, Test: {len(test_data)}")
print(f'train[0]: {train_data[0]}')
print(f'test[0]: {test_data[0]}')

if(DEBUG): input()

# Training 
iter = 0
NUM_EPOCHS = 4
for epoch in range(NUM_EPOCHS):
    for i in range(len(train_data)):

        # Read input prompts from train_data
        correct_ttl_bucket_index = train_data[i]['ttl_bucket']
        prompt = train_data[i]['prompt']
        responses = train_data[i]["tool_call_response"]
        tool_call_response = responses[epoch % len(responses)]

        # Get LLM response with tool call for given prompt
        LOG.debug(f'Prompt: {prompt}, Tool Call Response: {tool_call_response}')
        llm_response = llm_with_tool_call(prompt, tool_call_response)
        LOG.debug(f'LLM Response: {llm_response}')
        if(DEBUG): input()

        # Feature extraction
        feature_input = f'{prompt},{llm_response}'
        feature_vector = feature_extractor.extract(feature_input)
        features = feature_vector.as_dict()
        LOG.debug(feature_vector.as_dict())

        # LinUCB TTL classifier
        action, _score = policy.choose(features)
        propensity = policy.propensity(action, features)

        # LinUCB Online training
        reward = max(0, 1 - ((abs(correct_ttl_bucket_index - action)) / 3))

        # reward = 1.0 if action == correct_ttl_bucket_index else 0.0
        policy.update(action, reward, features)

        print(f'{i}: Correct TTL Bucket: {correct_ttl_bucket_index}, Predicted TTL Bucket: {action}, Reward: {reward}')
        LOG.debug('\n---------\n')

        # Snapshot periodically
        iter += 1
        if iter == 50:
            iter = 0
            print(f'epoch : {epoch} | index : {i} | Saving LinUCB policy snapshot to {snapshot_path}...')
            policy.save(snapshot_path)
# epoch : 4 | index : 1843 | Saving LinUCB policy snapshot to data/linucb_snapshot.json...

# Persist trained model
policy.save(snapshot_path)


# Testing
correct_predictions = {'0':0, '1':0, '2':0, '3':0}
total_predictions = {'0':0, '1':0, '2':0, '3':0}
for i in range(len(test_data)):

    # Get LLM response with tool call for given prompt
    correct_ttl_bucket_index = test_data[i]['ttl_bucket']
    prompt = test_data[i]['prompt']
    responses = train_data[i]["tool_call_response"]
    tool_call_response = responses[4 % len(responses)]

    llm_response = llm_with_tool_call(prompt, tool_call_response)

    # Feature extraction
    feature_input = f'{prompt},{llm_response}'
    feature_vector = feature_extractor.extract(feature_input)
    features = feature_vector.as_dict()

    # LinUCB TTL classifier
    action, _score = policy.choose(features)
    propensity = policy.propensity(action, features)

    if action == correct_ttl_bucket_index:
        correct_predictions[str(correct_ttl_bucket_index)] += 1
    total_predictions[str(correct_ttl_bucket_index)] += 1

    print(f'test-{i}: Correct TTL Bucket: {correct_ttl_bucket_index}, Predicted TTL Bucket: {action} | Correct predictions: {correct_predictions[str(correct_ttl_bucket_index)]} / Total predictions: {total_predictions[str(correct_ttl_bucket_index)]}')

print(f'Correct Predictions: {correct_predictions}')
print(f'Total Predictions: {total_predictions}')
print(f'Test Accuracy: {sum(correct_predictions.values())/len(test_data)*100}%')
