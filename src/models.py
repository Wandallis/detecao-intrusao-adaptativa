# src/models.py
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_random_forest_model(random_state=42):
    """Retorna o Random Forest configurado para o experimento."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )

def get_xgboost_model(scale_pos_weight=1.0, random_state=42):
    """Retorna o XGBoost configurado para o experimento."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )