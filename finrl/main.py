from processing import process
from training import train
from inference import get_inference
import sys


if __name__ == "__main__":
    try:
        '''
        Stage 1: processing.py
        '''
        # Get train.csv and trade.csv from period 2024-1-1 till 2 days before today
        process()
        print("Stage 1: Processing completed.")
        print("---------------------------------------------")
        
        '''
        Stage 2: training.py
        '''
        # Retrain the A2C model if necessary
        # Skipped for now, as the pre-trained A2C works well (2024-1-1 ~ 2025-4-14)
        # train()
        print("Stage 2: Training skipped.")
        print("---------------------------------------------")
        
        '''
        Stage 3: inference.py
        '''
        get_inference()
        print("Stage 3: Inference completed.")
        

    except Exception as e:
        print(f"Error in main occured -> {e}")
        sys.exit(1)