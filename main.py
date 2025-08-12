import logging
from prep_data import create_modeling_dataset

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    final_dataset = create_modeling_dataset()
