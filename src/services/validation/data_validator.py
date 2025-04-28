from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import numpy as np
from pydantic import BaseModel, ValidationError
import logging
from prometheus_client import Counter, Gauge
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    min_samples: int = 1000
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.1
    drift_threshold: float = 0.2
    batch_size: int = 100

class DataValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.baseline_statistics = {}
        self.scaler = StandardScaler()
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize validation metrics"""
        return {
            "validation_errors": Counter(
                "data_validation_errors_total",
                "Total validation errors",
                ["error_type", "field"]
            ),
            "data_drift": Gauge(
                "data_drift_score",
                "Data drift detection score",
                ["feature"]
            ),
            "quality_score": Gauge(
                "data_quality_score",
                "Overall data quality score",
                ["dataset"]
            )
        }
    
    async def validate_input(
        self,
        data: Dict,
        schema: BaseModel,
        context: Optional[Dict] = None
    ) -> tuple[bool, List[str]]:
        """Validate input data against schema"""
        try:
            # Basic schema validation
            try:
                validated_data = schema(**data)
            except ValidationError as e:
                errors = [f"{error['loc']}: {error['msg']}" for error in e.errors()]
                for error in errors:
                    self.metrics["validation_errors"].labels(
                        error_type="schema_validation",
                        field=error.split(":")[0]
                    ).inc()
                return False, errors
            
            # Additional validation checks
            errors = []
            
            # Check for missing values
            missing_errors = await self._check_missing_values(data)
            errors.extend(missing_errors)
            
            # Check for outliers
            outlier_errors = await self._check_outliers(data)
            errors.extend(outlier_errors)
            
            # Check for data drift if baseline exists
            if self.baseline_statistics:
                drift_errors = await self._check_data_drift(data)
                errors.extend(drift_errors)
            
            # Custom validation rules
            if context:
                custom_errors = await self._apply_custom_rules(data, context)
                errors.extend(custom_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, [str(e)]
    
    async def establish_baseline(self, data: List[Dict]):
        """Establish baseline statistics for drift detection"""
        try:
            df = pd.DataFrame(data)
            
            # Calculate baseline statistics
            self.baseline_statistics = {
                "numerical": {
                    col: {
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "quantiles": df[col].quantile([0.25, 0.5, 0.75]).to_dict()
                    }
                    for col in df.select_dtypes(include=[np.number]).columns
                },
                "categorical": {
                    col: df[col].value_counts(normalize=True).to_dict()
                    for col in df.select_dtypes(include=["object"]).columns
                }
            }
            
            # Fit scaler for numerical features
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if not numerical_cols.empty:
                self.scaler.fit(df[numerical_cols])
            
            logger.info("Established new baseline statistics")
            
        except Exception as e:
            logger.error(f"Baseline establishment failed: {e}")
            raise
    
    async def _check_missing_values(self, data: Dict) -> List[str]:
        """Check for missing values in data"""
        errors = []
        
        for key, value in data.items():
            if value is None:
                errors.append(f"Missing value for field: {key}")
                self.metrics["validation_errors"].labels(
                    error_type="missing_value",
                    field=key
                ).inc()
            elif isinstance(value, (str, list, dict)) and not value:
                errors.append(f"Empty value for field: {key}")
                self.metrics["validation_errors"].labels(
                    error_type="empty_value",
                    field=key
                ).inc()
        
        return errors
    
    async def _check_outliers(self, data: Dict) -> List[str]:
        """Check for outliers in numerical data"""
        errors = []
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if key in self.baseline_statistics.get("numerical", {}):
                    baseline = self.baseline_statistics["numerical"][key]
                    z_score = abs(
                        (value - baseline["mean"]) / baseline["std"]
                    )
                    
                    if z_score > self.config.outlier_threshold:
                        errors.append(
                            f"Outlier detected in field {key}: "
                            f"value {value} (z-score: {z_score:.2f})"
                        )
                        self.metrics["validation_errors"].labels(
                            error_type="outlier",
                            field=key
                        ).inc()
        
        return errors
    
    async def _check_data_drift(self, data: Dict) -> List[str]:
        """Check for data drift against baseline"""
        errors = []
        
        # Check numerical features
        for key, value in data.items():
            if key in self.baseline_statistics.get("numerical", {}):
                baseline = self.baseline_statistics["numerical"][key]
                
                # Calculate drift score using KS statistic
                current_value = np.array([[value]])
                scaled_value = self.scaler.transform(current_value)[0][0]
                
                drift_score = abs(
                    (scaled_value - baseline["mean"]) / baseline["std"]
                )
                
                self.metrics["data_drift"].labels(
                    feature=key
                ).set(drift_score)
                
                if drift_score > self.config.drift_threshold:
                    errors.append(
                        f"Data drift detected in field {key}: "
                        f"drift score {drift_score:.2f}"
                    )
            
            # Check categorical features
            elif key in self.baseline_statistics.get("categorical", {}):
                baseline_dist = self.baseline_statistics["categorical"][key]
                if value not in baseline_dist:
                    errors.append(
                        f"New category detected in field {key}: {value}"
                    )
        
        return errors
    
    async def _apply_custom_rules(
        self,
        data: Dict,
        context: Dict
    ) -> List[str]:
        """Apply custom validation rules"""
        errors = []
        
        # Example custom rules
        if "timestamp" in data:
            # Check if timestamp is in future
            try:
                data_time = datetime.fromisoformat(data["timestamp"])
                if data_time > self.timestamp:
                    errors.append("Timestamp cannot be in future")
            except ValueError:
                errors.append("Invalid timestamp format")
        
        if "amount" in data and "currency" in data:
            # Check currency-specific amount limits
            currency_limits = context.get("currency_limits", {})
            if data["currency"] in currency_limits:
                max_amount = currency_limits[data["currency"]]
                if data["amount"] > max_amount:
                    errors.append(
                        f"Amount exceeds limit for {data['currency']}: "
                        f"{data['amount']} > {max_amount}"
                    )
        
        return errors
    
    async def calculate_quality_score(self, data: Dict) -> float:
        """Calculate overall quality score for data"""
        try:
            scores = []
            
            # Check completeness
            total_fields = len(data)
            missing_fields = sum(1 for v in data.values() if v is None)
            completeness_score = 1 - (missing_fields / total_fields)
            scores.append(completeness_score)
            
            # Check validity
            valid_count = 0
            for key, value in data.items():
                if key in self.baseline_statistics.get("numerical", {}):
                    baseline = self.baseline_statistics["numerical"][key]
                    z_score = abs((value - baseline["mean"]) / baseline["std"])
                    if z_score <= self.config.outlier_threshold:
                        valid_count += 1
                elif key in self.baseline_statistics.get("categorical", {}):
                    if value in self.baseline_statistics["categorical"][key]:
                        valid_count += 1
            
            validity_score = valid_count / total_fields
            scores.append(validity_score)
            
            # Calculate final score
            quality_score = np.mean(scores)
            
            # Update metric
            self.metrics["quality_score"].labels(
                dataset="current"
            ).set(quality_score)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0