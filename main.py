import logging
from prep_data import create_modeling_dataset
from rnn_modeling import LSTM_model
from USDA_model import USDA_model
from config import Config

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    # Get Yield Data
    final_dataset = create_modeling_dataset()

    #Yield Prediction using LSTM model
    LSTM_model(Config.Paths.OUTPUT_DATA)

    # Yield Prediction using USDA model
    USDA_model(Config.Paths.OUTPUT_DATA)


