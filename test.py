import logging

logger = logging.getLogger()
# logger.disabled = True

log_format = "%(asctime)s | %(levelname)s: %(message)s"
logging.basicConfig(
    filename="test_logfile.log",
    format=log_format,
    level=logging.DEBUG,
)

logging.debug("debug")
logging.info("info")
logging.warning("warning")
print("output")
