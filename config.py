import torch

# Paths
RAW_DATA_CSV = 'news.csv'
TEMP_PROCESSED_JSON = 'temp_processed_data.json'
NEWS_WITH_SCORE_CSV = 'news_with_risk_score.csv'
TEMP_DATE_RISK_CSV = 'temp_date_risk.csv'
AGGREGATED_WEIGHTS_CSV = 'aggregated_risk_scores.csv'

# Model
G_LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# SOURCE WEIGHTS
WEIGHT_NEWS = 0.5707
WEIGHT_REDDIT = 0.4293

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"