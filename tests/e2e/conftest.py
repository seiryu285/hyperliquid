"""
E2E test configuration and fixtures.
"""

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import subprocess
import time
import requests
from typing import Generator

@pytest.fixture(scope="session")
def setup_test_env():
    """Setup test environment."""
    # Start backend server
    backend = subprocess.Popen(
        ["uvicorn", "backend.main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start frontend server
    frontend = subprocess.Popen(
        ["npm", "start", "--prefix", "frontend"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for servers to start
    time.sleep(5)
    
    # Verify servers are running
    try:
        requests.get("http://localhost:8000/health")
        requests.get("http://localhost:3000")
    except requests.exceptions.ConnectionError:
        backend.kill()
        frontend.kill()
        pytest.fail("Failed to start test environment")
    
    yield
    
    # Cleanup
    backend.kill()
    frontend.kill()

@pytest.fixture
def driver() -> Generator[webdriver.Chrome, None, None]:
    """Setup Chrome WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(10)
    
    yield driver
    
    driver.quit()

@pytest.fixture
def base_url() -> str:
    """Get base URL for frontend application."""
    return "http://localhost:3000"
