"""
OWASP ZAP security scan script.
"""

import time
import sys
from zapv2 import ZAPv2
import json
from datetime import datetime

class SecurityScanner:
    """Security scanner using OWASP ZAP."""
    
    def __init__(self, target_url: str, api_key: str = None):
        """Initialize scanner."""
        self.target_url = target_url
        self.zap = ZAPv2(
            apikey=api_key,
            proxies={'http': 'http://127.0.0.1:8080', 'https': 'http://127.0.0.1:8080'}
        )
    
    def setup(self):
        """Setup scanning environment."""
        print('Starting ZAP session...')
        self.zap.core.new_session()
        
        # Configure session properties
        self.zap.core.set_mode('attack')
        
        # Include target in scope
        self.zap.context.include_in_context('Default Context', f'^{self.target_url}.*$')
        
        # Configure authentication
        self.zap.authentication.set_authentication_method(
            'Default Context',
            'formBasedAuthentication',
            'loginUrl=' + self.target_url + '/login' +
            '&loginRequestData=email={%email%}&password={%password%}'
        )
        
        # Set logged in/out indicators
        self.zap.authentication.set_logged_in_indicator('Default Context', '\Q<div id="dashboard">\E')
        self.zap.authentication.set_logged_out_indicator('Default Context', '\Q<div id="login">\E')
    
    def spider_scan(self):
        """Run spider scan."""
        print('Starting Spider scan...')
        scan_id = self.zap.spider.scan(self.target_url)
        
        # Wait for Spider scan to complete
        while int(self.zap.spider.status(scan_id)) < 100:
            print(f'Spider progress: {self.zap.spider.status(scan_id)}%')
            time.sleep(5)
        
        print('Spider scan completed')
    
    def active_scan(self):
        """Run active scan."""
        print('Starting Active scan...')
        scan_id = self.zap.ascan.scan(self.target_url)
        
        # Wait for Active scan to complete
        while int(self.zap.ascan.status(scan_id)) < 100:
            print(f'Active scan progress: {self.zap.ascan.status(scan_id)}%')
            time.sleep(5)
        
        print('Active scan completed')
    
    def generate_report(self):
        """Generate security report."""
        print('Generating report...')
        
        # Get all alerts
        alerts = self.zap.core.alerts()
        
        # Organize alerts by risk level
        report = {
            'scan_date': datetime.now().isoformat(),
            'target_url': self.target_url,
            'high_risks': [],
            'medium_risks': [],
            'low_risks': [],
            'informational': []
        }
        
        for alert in alerts:
            alert_dict = {
                'alert': alert.get('alert'),
                'risk': alert.get('risk'),
                'reliability': alert.get('reliability'),
                'url': alert.get('url'),
                'param': alert.get('param'),
                'evidence': alert.get('evidence'),
                'solution': alert.get('solution')
            }
            
            if alert.get('risk') == 'High':
                report['high_risks'].append(alert_dict)
            elif alert.get('risk') == 'Medium':
                report['medium_risks'].append(alert_dict)
            elif alert.get('risk') == 'Low':
                report['low_risks'].append(alert_dict)
            else:
                report['informational'].append(alert_dict)
        
        # Save report
        filename = f'security_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f'Report saved as {filename}')
        
        # Print summary
        print('\nScan Summary:')
        print(f'High Risk Issues: {len(report["high_risks"])}')
        print(f'Medium Risk Issues: {len(report["medium_risks"])}')
        print(f'Low Risk Issues: {len(report["low_risks"])}')
        print(f'Informational: {len(report["informational"])}')
        
        return report

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print('Usage: python zap_scan.py <target_url> [api_key]')
        sys.exit(1)
    
    target_url = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    scanner = SecurityScanner(target_url, api_key)
    
    try:
        scanner.setup()
        scanner.spider_scan()
        scanner.active_scan()
        scanner.generate_report()
    except Exception as e:
        print(f'Error during security scan: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()
