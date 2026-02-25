"""
Configuration management for the Autonomous RL Trading Agent.
All configurations are validated with Pydantic for type safety.
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseSettings, Field, validator
import structlog

logger = structlog.get_logger(__name__)

class TradingConfig(BaseSettings):
    """Trading system configuration"""
    
    # Agent Architecture
    ENABLED_AGENTS: List[str] = Field(
        default=["PPO", "SAC", "DDPG"],
        description="Specialist agents to activate"
    )
    
    # Training Parameters
    TRAINING_EPISODES: int = Field(default=1000, ge=100, le=10000)
    WARMUP_STEPS: int = Field(default=1000, description="Steps before meta-learning begins")
    BATCH_SIZE: int = Field(default=64, ge=32, le=1024)
    
    # Risk Management
    MAX_POSITION_SIZE: float = Field(default=0.1, ge=0.01, le=0.5)
    STOP_LOSS_PERCENT: float = Field(default=0.02, ge=0.005, le=0.1)
    MAX_DRAWDOWN_LIMIT: float = Field(default=0.15, ge=0.05, le=0.3)
    
    # Market Parameters
    SYMBOLS: List[str] = Field(default=["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    TIMEFRAME: str = Field(default="5m", regex="^(1m|5m|15m|1h|4h|1d)$")
    LOOKBACK_WINDOW: int = Field(default=50, ge=20, le=200)
    
    # Ensemble Parameters
    REGIME_DETECTION_HORIZON: int = Field(default=20, description="Lookback for market regime detection")
    WEIGHT_UPDATE_FREQUENCY: int = Field(default=100, description="Steps between ensemble rebalancing")
    
    # Firebase Configuration
    FIREBASE_PROJECT_ID: Optional[str] = None
    FIREBASE_CREDENTIALS_PATH: Optional[str] = "./firebase-credentials.json"
    
    # Exchange Configuration
    EXCHANGE_ID: str = Field(default="binance", regex="^(binance|coinbase|kraken)$")
    PAPER_TRADING: bool = Field(default=True)
    
    class Config:
        env_prefix = "TRADING_"
        case_sensitive = False
        env_file = ".env"
        
    @validator("SYMBOLS")
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one trading symbol must be configured")
        return v
    
    @validator("ENABLED_AGENTS")
    def validate_agents(cls, v):
        valid_agents = ["PPO", "SAC", "DDPG", "A2C", "TD3"]
        for agent in v:
            if agent not in valid_agents:
                raise ValueError(f"Invalid agent type: {agent}. Valid: {valid_agents}")
        return v

class MetaCognitiveConfig(BaseSettings):
    """Meta-cognitive controller configuration"""
    
    REGIME_CLUSTERS: int = Field(default=3, ge=2, le=5, description="Number of market regimes")
    ENSEMBLE_MEMORY_SIZE: int = Field(default=1000, description="Size of performance memory buffer")
    PERFORMANCE_HALF_LIFE: float = Field(default=100, description="Half-life for performance decay")
    
    # Regime Detection Parameters
    VOLATILITY_THRESHOLD: float = Field(default=0.02)
    TREND_THRESHOLD: float = Field(default=0.001)
    CORRELATION_THRESHOLD: float = Field(default=0.7)
    
    # Learning Parameters
    META_LEARNING_RATE: float = Field(default=0.001)
    EXPLORATION_RATE: float = Field(default=0.1, ge=0, le=1)
    
    class Config:
        env_prefix = "META_"
        env_file = ".env"

# Global configuration instances
trading_config = TradingConfig()
meta_config = MetaCognitiveConfig()

def validate_configuration():
    """Validate and log configuration"""
    logger.info(
        "Configuration validated",
        enabled_agents=trading_config.ENABLED_AGENTS,
        symbols=trading_config.SYMBOLS,
        max_position=trading_config.MAX_POSITION_SIZE,
        paper_trading=trading_config.PAPER_TRADING
    )
    
    # Validate Firebase configuration if not in paper trading
    if not trading_config.PAPER_TRADING and not trading_config.FIREBASE_PROJECT_ID:
        logger.error("Firebase configuration required for live trading")
        raise ValueError("FIREBASE_PROJECT_ID must be set for live trading")
    
    return True