"""
Custom alert template system with Jinja2-based templating.
"""

from jinja2 import Template
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Default templates for different alert types
DEFAULT_TEMPLATES = {
    "security": {
        "email": """
Security Alert: {{ alert.name }}
Severity: {{ alert.severity }}
Time: {{ alert.timestamp }}

Description: {{ alert.description }}

Details:
{% for key, value in alert.details.items() %}
- {{ key }}: {{ value }}
{% endfor %}

{% if alert.recommendations %}
Recommendations:
{% for rec in alert.recommendations %}
- {{ rec }}
{% endfor %}
{% endif %}

Dashboard: {{ dashboard_url }}
        """.strip(),
        
        "slack": """
:warning: *Security Alert: {{ alert.name }}*
*Severity*: {{ alert.severity }}
*Time*: {{ alert.timestamp }}

*Description*: {{ alert.description }}

*Details*:
{% for key, value in alert.details.items() %}
• {{ key }}: {{ value }}
{% endfor %}

{% if alert.recommendations %}
*Recommendations*:
{% for rec in alert.recommendations %}
• {{ rec }}
{% endfor %}
{% endif %}

<{{ dashboard_url }}|View Dashboard>
        """.strip(),
        
        "webhook": {
            "title": "Security Alert: {{ alert.name }}",
            "severity": "{{ alert.severity }}",
            "timestamp": "{{ alert.timestamp }}",
            "description": "{{ alert.description }}",
            "details": "{{ alert.details | tojson }}",
            "recommendations": "{{ alert.recommendations | tojson if alert.recommendations }}",
            "dashboard_url": "{{ dashboard_url }}"
        }
    },
    
    "performance": {
        "email": """
Performance Alert: {{ alert.name }}
Severity: {{ alert.severity }}
Time: {{ alert.timestamp }}

Metric: {{ alert.metric }}
Current Value: {{ alert.current_value }}
Threshold: {{ alert.threshold }}

Description: {{ alert.description }}

Impact:
{% for item in alert.impact %}
- {{ item }}
{% endfor %}

{% if alert.recommendations %}
Recommendations:
{% for rec in alert.recommendations %}
- {{ rec }}
{% endfor %}
{% endif %}

Dashboard: {{ dashboard_url }}
        """.strip(),
        
        "slack": """
:fire: *Performance Alert: {{ alert.name }}*
*Severity*: {{ alert.severity }}
*Time*: {{ alert.timestamp }}

*Metric*: {{ alert.metric }}
*Current Value*: {{ alert.current_value }}
*Threshold*: {{ alert.threshold }}

*Description*: {{ alert.description }}

*Impact*:
{% for item in alert.impact %}
• {{ item }}
{% endfor %}

{% if alert.recommendations %}
*Recommendations*:
{% for rec in alert.recommendations %}
• {{ rec }}
{% endfor %}
{% endif %}

<{{ dashboard_url }}|View Dashboard>
        """.strip(),
        
        "webhook": {
            "title": "Performance Alert: {{ alert.name }}",
            "severity": "{{ alert.severity }}",
            "timestamp": "{{ alert.timestamp }}",
            "metric": "{{ alert.metric }}",
            "current_value": "{{ alert.current_value }}",
            "threshold": "{{ alert.threshold }}",
            "description": "{{ alert.description }}",
            "impact": "{{ alert.impact | tojson }}",
            "recommendations": "{{ alert.recommendations | tojson if alert.recommendations }}",
            "dashboard_url": "{{ dashboard_url }}"
        }
    },
    
    "anomaly": {
        "email": """
Anomaly Alert: {{ alert.name }}
Severity: {{ alert.severity }}
Time: {{ alert.timestamp }}

Anomaly Score: {{ alert.score }}
Affected Metrics:
{% for metric, value in alert.metrics.items() %}
- {{ metric }}: {{ value }}
{% endfor %}

Description: {{ alert.description }}

Pattern Analysis:
{% for pattern in alert.patterns %}
- {{ pattern }}
{% endfor %}

{% if alert.recommendations %}
Recommendations:
{% for rec in alert.recommendations %}
- {{ rec }}
{% endfor %}
{% endif %}

Dashboard: {{ dashboard_url }}
        """.strip(),
        
        "slack": """
:rotating_light: *Anomaly Alert: {{ alert.name }}*
*Severity*: {{ alert.severity }}
*Time*: {{ alert.timestamp }}

*Anomaly Score*: {{ alert.score }}
*Affected Metrics*:
{% for metric, value in alert.metrics.items() %}
• {{ metric }}: {{ value }}
{% endfor %}

*Description*: {{ alert.description }}

*Pattern Analysis*:
{% for pattern in alert.patterns %}
• {{ pattern }}
{% endfor %}

{% if alert.recommendations %}
*Recommendations*:
{% for rec in alert.recommendations %}
• {{ rec }}
{% endfor %}
{% endif %}

<{{ dashboard_url }}|View Dashboard>
        """.strip(),
        
        "webhook": {
            "title": "Anomaly Alert: {{ alert.name }}",
            "severity": "{{ alert.severity }}",
            "timestamp": "{{ alert.timestamp }}",
            "score": "{{ alert.score }}",
            "metrics": "{{ alert.metrics | tojson }}",
            "description": "{{ alert.description }}",
            "patterns": "{{ alert.patterns | tojson }}",
            "recommendations": "{{ alert.recommendations | tojson if alert.recommendations }}",
            "dashboard_url": "{{ dashboard_url }}"
        }
    }
}

class AlertTemplateManager:
    """Manager for alert templates."""
    
    def __init__(self):
        """Initialize template manager with default templates."""
        self.templates = DEFAULT_TEMPLATES
        self.custom_templates = {}
    
    def add_custom_template(
        self,
        alert_type: str,
        channel: str,
        template: str
    ) -> None:
        """Add or update custom template."""
        if alert_type not in self.custom_templates:
            self.custom_templates[alert_type] = {}
        self.custom_templates[alert_type][channel] = template
    
    def get_template(
        self,
        alert_type: str,
        channel: str
    ) -> Optional[str]:
        """Get template for alert type and channel."""
        # Check custom templates first
        if (alert_type in self.custom_templates and
            channel in self.custom_templates[alert_type]):
            return self.custom_templates[alert_type][channel]
        
        # Fall back to default templates
        if alert_type in self.templates and channel in self.templates[alert_type]:
            return self.templates[alert_type][channel]
        
        return None
    
    def render_alert(
        self,
        alert_type: str,
        channel: str,
        alert_data: Dict[str, Any],
        dashboard_url: str
    ) -> Optional[str]:
        """Render alert using template."""
        try:
            template_str = self.get_template(alert_type, channel)
            if not template_str:
                logger.error(
                    f"No template found for {alert_type} and {channel}"
                )
                return None
            
            # Create template
            template = Template(template_str)
            
            # Add common context
            context = {
                "alert": alert_data,
                "dashboard_url": dashboard_url,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Render template
            if channel == "webhook":
                # For webhooks, return JSON
                rendered = template.render(**context)
                return json.dumps(json.loads(rendered))
            else:
                # For email and Slack, return formatted text
                return template.render(**context)
            
        except Exception as e:
            logger.error(f"Error rendering alert template: {e}")
            return None
    
    def validate_template(
        self,
        template_str: str,
        sample_data: Dict[str, Any]
    ) -> bool:
        """Validate template with sample data."""
        try:
            template = Template(template_str)
            template.render(**sample_data)
            return True
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False

# Global template manager instance
template_manager = AlertTemplateManager()
