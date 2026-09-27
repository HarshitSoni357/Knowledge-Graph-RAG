from enum import Enum


class EntityType(str, Enum):
    COMPANY = "Company"
    PERSON = "Person"
    PRODUCT = "Product"
    BUSINESS_SEGMENT = "BusinessSegment"
    INDUSTRY = "Industry"
    LOCATION = "Location"
    FINANCIAL_METRIC = "FinancialMetric"
    EVENT = "Event"


class RelationshipType(str, Enum):
    ACQUIRED = "ACQUIRED"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    FOUNDED_BY = "FOUNDED_BY"
    LED_BY = "LED_BY"
    EXECUTIVE_OF = "EXECUTIVE_OF"
    OWNS = "OWNS"
    PRODUCES = "PRODUCES"
    OPERATES_IN = "OPERATES_IN"
    COMPETES_WITH = "COMPETES_WITH"
    PART_OF = "PART_OF"
    REPORTED_METRIC = "REPORTED_METRIC"
    RELATED_TO = "RELATED_TO"
    OCCURRED_IN = "OCCURRED_IN"