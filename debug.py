import json, os 
from config import LABELED_PATH, HIDDEN_STATES_DIR 
print('LABELED_PATH:', LABELED_PATH) 
print('EXISTS:', os.path.exists(LABELED_PATH)) 
print('HIDDEN_STATES_DIR:', HIDDEN_STATES_DIR) 
print('DIR EXISTS:', os.path.exists(HIDDEN_STATES_DIR)) 
f = open(LABELED_PATH) 
records = json.load(f) 
print('Records:', len(records)) 
