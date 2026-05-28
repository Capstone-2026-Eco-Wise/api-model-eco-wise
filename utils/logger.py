import logging

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

logger = logging.getLogger("ecowise-api")
