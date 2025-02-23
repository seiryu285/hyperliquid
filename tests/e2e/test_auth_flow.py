"""
E2E tests for authentication flows.
"""

import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pyotp

class TestAuthFlow:
    """Test authentication flows."""
    
    def test_successful_registration(self, driver, base_url):
        """Test successful user registration."""
        driver.get(f"{base_url}/register")
        
        # Fill registration form
        driver.find_element(By.ID, "email").send_keys("test@example.com")
        driver.find_element(By.ID, "username").send_keys("testuser")
        driver.find_element(By.ID, "password").send_keys("Test123!@#")
        driver.find_element(By.ID, "confirm-password").send_keys("Test123!@#")
        
        # Submit form
        driver.find_element(By.ID, "register-button").click()
        
        # Wait for success message
        success_message = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
        )
        assert "Registration successful" in success_message.text
    
    def test_login_with_2fa(self, driver, base_url):
        """Test login flow with 2FA."""
        driver.get(f"{base_url}/login")
        
        # Login
        driver.find_element(By.ID, "email").send_keys("test@example.com")
        driver.find_element(By.ID, "password").send_keys("Test123!@#")
        driver.find_element(By.ID, "login-button").click()
        
        # Setup 2FA if not already setup
        try:
            setup_2fa = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, "setup-2fa-button"))
            )
            setup_2fa.click()
            
            # Store TOTP secret
            totp_secret = driver.find_element(By.ID, "totp-secret").text
            totp = pyotp.TOTP(totp_secret)
            
            # Enter TOTP code
            driver.find_element(By.ID, "totp-code").send_keys(totp.now())
            driver.find_element(By.ID, "verify-2fa-button").click()
        except TimeoutException:
            # 2FA already setup
            pass
        
        # Enter 2FA code
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "2fa-input"))
        )
        totp = pyotp.TOTP(totp_secret)
        driver.find_element(By.ID, "2fa-input").send_keys(totp.now())
        driver.find_element(By.ID, "verify-2fa-button").click()
        
        # Verify successful login
        dashboard = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "dashboard"))
        )
        assert dashboard.is_displayed()
    
    def test_password_reset(self, driver, base_url):
        """Test password reset flow."""
        driver.get(f"{base_url}/forgot-password")
        
        # Request password reset
        driver.find_element(By.ID, "email").send_keys("test@example.com")
        driver.find_element(By.ID, "reset-button").click()
        
        # Verify success message
        success_message = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
        )
        assert "Password reset instructions sent" in success_message.text
    
    def test_rate_limiting(self, driver, base_url):
        """Test rate limiting on login attempts."""
        driver.get(f"{base_url}/login")
        
        # Attempt multiple failed logins
        for _ in range(6):
            driver.find_element(By.ID, "email").send_keys("test@example.com")
            driver.find_element(By.ID, "password").send_keys("wrong_password")
            driver.find_element(By.ID, "login-button").click()
            
            try:
                error_message = WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "error-message"))
                )
                if "Rate limit exceeded" in error_message.text:
                    break
            except TimeoutException:
                continue
        
        # Verify rate limit message
        error_message = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "error-message"))
        )
        assert "Rate limit exceeded" in error_message.text
    
    def test_brute_force_protection(self, driver, base_url):
        """Test brute force protection."""
        driver.get(f"{base_url}/login")
        
        # Attempt multiple failed logins
        for _ in range(6):
            driver.find_element(By.ID, "email").send_keys("test@example.com")
            driver.find_element(By.ID, "password").send_keys("wrong_password")
            driver.find_element(By.ID, "login-button").click()
            
            try:
                error_message = WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "error-message"))
                )
                if "Account locked" in error_message.text:
                    break
            except TimeoutException:
                continue
        
        # Verify account lockout message
        error_message = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "error-message"))
        )
        assert "Account locked" in error_message.text
