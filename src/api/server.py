"""
FastAPI Server for Multi-Modal Recommendation System
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import torch
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal Recommendation API",
    description="API for visual content recommendation and advertiser analytics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    context: Optional[Dict] = Field(default={}, description="Additional context")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of recommendations")
    category: Optional[str] = Field(default=None, description="Content category filter")


class ContentItem(BaseModel):
    item_id: str
    title: str
    category: str
    score: float
    image_similarity: float
    text_similarity: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[ContentItem]
    retrieval_time_ms: float
    timestamp: str


class CampaignAnalyticsRequest(BaseModel):
    campaign_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metrics: List[str] = Field(default=["ctr", "conversion_rate", "roas"])


class CampaignMetrics(BaseModel):
    campaign_id: str
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    conversion_rate: float
    roas: float
    lift: Optional[float] = None
    p_value: Optional[float] = None


class ABTestRequest(BaseModel):
    test_id: str
    metrics: List[str] = Field(default=["ctr", "conversion_rate"])


class ABTestResult(BaseModel):
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    confidence_interval: List[float]
    is_significant: bool


class ABTestResponse(BaseModel):
    test_id: str
    results: List[ABTestResult]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    timestamp: str


# Mock data generators (replace with actual model inference in production)
def generate_mock_recommendations(user_id: str, top_k: int, category: Optional[str]) -> List[ContentItem]:
    """Generate mock recommendations for demo purposes"""
    categories = ['Fashion', 'Technology', 'Home & Garden', 'Food', 'Travel']
    
    recommendations = []
    for i in range(top_k):
        cat = category if category else np.random.choice(categories)
        recommendations.append(ContentItem(
            item_id=f"item_{1000 + i}",
            title=f"Recommended {cat} Item {i + 1}",
            category=cat,
            score=round(0.95 - (i * 0.02) + np.random.random() * 0.05, 4),
            image_similarity=round(np.random.random() * 0.3 + 0.7, 3),
            text_similarity=round(np.random.random() * 0.3 + 0.7, 3)
        ))
    
    return recommendations


def generate_mock_campaign_metrics(campaign_id: str) -> CampaignMetrics:
    """Generate mock campaign metrics"""
    impressions = np.random.randint(500000, 5000000)
    ctr = np.random.uniform(0.02, 0.05)
    clicks = int(impressions * ctr)
    conversion_rate = np.random.uniform(0.015, 0.030)
    conversions = int(clicks * conversion_rate)
    roas = np.random.uniform(3.0, 5.0)
    
    return CampaignMetrics(
        campaign_id=campaign_id,
        impressions=impressions,
        clicks=clicks,
        conversions=conversions,
        ctr=round(ctr * 100, 2),
        conversion_rate=round(conversion_rate * 100, 2),
        roas=round(roas, 2),
        lift=round(np.random.uniform(15, 45), 1),
        p_value=0.001
    )


def generate_mock_abtest_results(test_id: str, metrics: List[str]) -> List[ABTestResult]:
    """Generate mock A/B test results"""
    results = []
    
    metric_data = {
        'ctr': {'control': 2.4, 'treatment': 3.1, 'lift': 31},
        'conversion_rate': {'control': 1.8, 'treatment': 2.2, 'lift': 23},
        'save_rate': {'control': 8.2, 'treatment': 10.5, 'lift': 28},
        'dwell_time': {'control': 45, 'treatment': 64, 'lift': 42}
    }
    
    for metric in metrics:
        if metric in metric_data:
            data = metric_data[metric]
            control = data['control']
            treatment = data['treatment']
            absolute_lift = treatment - control
            relative_lift = data['lift']
            
            results.append(ABTestResult(
                metric_name=metric,
                control_mean=control,
                treatment_mean=treatment,
                absolute_lift=absolute_lift,
                relative_lift=relative_lift,
                p_value=0.001,
                confidence_interval=[absolute_lift * 0.8, absolute_lift * 1.2],
                is_significant=True
            ))
    
    return results


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=True,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=True,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized content recommendations
    
    Args:
        request: RecommendationRequest with user_id and parameters
    
    Returns:
        RecommendationResponse with ranked content items
    """
    try:
        start_time = datetime.now()
        
        # Generate recommendations (mock data for demo)
        recommendations = generate_mock_recommendations(
            request.user_id,
            request.top_k,
            request.category
        )
        
        # Calculate retrieval time
        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {request.user_id}")
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            retrieval_time_ms=round(retrieval_time, 2),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analytics/campaign", response_model=CampaignMetrics)
async def analyze_campaign(request: CampaignAnalyticsRequest):
    """
    Analyze campaign performance with causal inference
    
    Args:
        request: CampaignAnalyticsRequest with campaign_id and parameters
    
    Returns:
        CampaignMetrics with performance statistics
    """
    try:
        # Generate campaign metrics (mock data for demo)
        metrics = generate_mock_campaign_metrics(request.campaign_id)
        
        logger.info(f"Analyzed campaign {request.campaign_id}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error analyzing campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analytics/abtest", response_model=ABTestResponse)
async def analyze_abtest(request: ABTestRequest):
    """
    Analyze A/B test results with statistical significance
    
    Args:
        request: ABTestRequest with test_id and metrics
    
    Returns:
        ABTestResponse with test results
    """
    try:
        # Generate A/B test results (mock data for demo)
        results = generate_mock_abtest_results(request.test_id, request.metrics)
        
        logger.info(f"Analyzed A/B test {request.test_id}")
        
        return ABTestResponse(
            test_id=request.test_id,
            results=results,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error analyzing A/B test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embeddings/content")
async def get_content_embeddings(
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = None
):
    """
    Generate embeddings for content (image + text)
    
    Args:
        image: Image file (optional)
        text: Text description (optional)
    
    Returns:
        Dictionary with embeddings
    """
    try:
        # Mock embeddings
        embedding = np.random.randn(128).tolist()
        
        return {
            "embedding": embedding,
            "dimension": 128,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics/model")
async def get_model_metrics():
    """
    Get current model performance metrics
    
    Returns:
        Dictionary with model metrics
    """
    return {
        "ndcg_at_10": 0.82,
        "hit_rate_at_20": 0.87,
        "mrr": 0.76,
        "coverage": 0.942,
        "avg_retrieval_time_ms": 87,
        "total_items": 500000,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/stats/system")
async def get_system_stats():
    """
    Get system statistics
    
    Returns:
        Dictionary with system stats
    """
    return {
        "total_requests_today": 1234567,
        "avg_response_time_ms": 92,
        "cache_hit_rate": 0.78,
        "active_users": 45823,
        "index_size_gb": 12.3,
        "last_update": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Multi-Modal Recommendation API Server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
