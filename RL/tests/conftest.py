import logging
import datetime
import pytest
import utils.custom_paths as custom_paths

# Używamy hooka sesji, aby zapisać start sesji przez logger
def pytest_sessionstart(session):
    # To wyśle informację do pliku log_file zdefiniowanego w toml
    logging.info(f"\n\n{'#'*30}")
    logging.info(f"NOWA SESJA TESTOWA: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"{'#'*30}\n")

def pytest_runtest_setup(item):
    """Wykonuje się przed każdym testem"""
    logging.info(f"==== START TEST: {item.nodeid} ====")

def pytest_runtest_teardown(item, nextitem):
    """Wykonuje się po każdym teście"""
    logging.info(f"---- KONIEC TESTU: {item.name} ----\n")