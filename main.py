from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io
import torch
from torchvision import transforms
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
from datetime import datetime
import uvicorn
from deepgaze_pytorch import DeepGazeIIE
import plotly.express as px
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Attention Analysis API",
    description="API for analyzing visual attention patterns in images",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create thread pool for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# In-memory cache with TTL
analysis_cache = {}

def is_blank_image(image_array: np.ndarray, threshold: float = 0.99) -> bool:
    """Check if an image is blank or nearly blank"""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
        
    white_ratio = np.mean(gray >= 250)
    black_ratio = np.mean(gray <= 5)
    variance = np.var(gray)
    std_dev = np.std(gray)
    
    return (white_ratio >= threshold or
            black_ratio >= threshold or
            variance < 2.0 or
            std_dev < 1.5)

class EnhancedAttentionAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepGazeIIE()
        self.model.to(self.device)
        self.model.eval()
        self.init_transform()
        self.init_adaptive_benchmarks()
        logger.info(f"Initialized EnhancedAttentionAnalyzer using device: {self.device}")

    def init_transform(self):
        """Initialize image transformation pipeline"""
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def init_adaptive_benchmarks(self):
        """Initialize adaptive benchmark system"""
        self.benchmarks = {
            'cognitive_demand': {
                'low': {'base': 30, 'scale': 0.8},
                'medium': {'base': 60, 'scale': 1.0},
                'high': {'base': 90, 'scale': 1.2}
            },
            'focus_score': {
                'low': {'base': 50, 'scale': 0.8},
                'medium': {'base': 70, 'scale': 1.0},
                'high': {'base': 85, 'scale': 1.2}
            },
            'clarity_score': {
                'low': {'base': 60, 'scale': 0.8},
                'medium': {'base': 75, 'scale': 1.0},
                'high': {'base': 90, 'scale': 1.2}
            },
            'engagement_score': {
                'low': {'base': 40, 'scale': 0.8},
                'medium': {'base': 65, 'scale': 1.0},
                'high': {'base': 80, 'scale': 1.2}
            }
        }
        self.benchmark_history = []

    def determine_optimal_size(self, image: Image.Image) -> Tuple[int, int]:
        """Determine optimal size for processing"""
        orig_width, orig_height = image.size
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        min_dim, max_dim = 224, 448
        complexity_scale = np.clip(laplacian_var / 500, 0.5, 2.0)
        
        if orig_width > orig_height:
            target_width = int(min(max_dim, max(min_dim, orig_width * complexity_scale)))
            target_height = int(target_width * orig_height / orig_width)
        else:
            target_height = int(min(max_dim, max(min_dim, orig_height * complexity_scale)))
            target_width = int(target_height * orig_width / orig_height)
            
        return target_width, target_height

    def create_dynamic_centerbias(self, image_size: Tuple[int, int], content_analysis: Dict) -> torch.Tensor:
        """Create dynamic center bias based on content"""
        h, w = image_size
        y = torch.linspace(-1, 1, h)
        x = torch.linspace(-1, 1, w)
        grid_y, grid_x = torch.meshgrid(y, x)
        d = torch.sqrt(grid_x**2 + grid_y**2)
        
        base_sigma = 0.5
        complexity_factor = content_analysis.get('complexity', 1.0)
        size_factor = min(h, w) / 224.0
        sigma = np.clip(base_sigma * complexity_factor * size_factor, 0.3, 0.8)
        
        return torch.exp(-d**2 / (2 * sigma**2)).unsqueeze(0).to(self.device)

    def analyze_image_content(self, image: Image.Image) -> Dict:
        """Analyze image content for parameter adjustment"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        
        return {
            'complexity': np.clip(laplacian_var / 500, 0.5, 2.0),
            'edge_density': edge_density,
            'contrast': contrast
        }

    def predict_saliency(self, image: Image.Image) -> np.ndarray:
        """Generate saliency map"""
        try:
            content_analysis = self.analyze_image_content(image)
            target_width, target_height = self.determine_optimal_size(image)
            
            dynamic_transform = transforms.Compose([
                transforms.Resize((target_height, target_width)),
                self.base_transform
            ])
            
            img_tensor = dynamic_transform(image).unsqueeze(0).to(self.device)
            center_bias = self.create_dynamic_centerbias((target_height, target_width), content_analysis)
            
            with torch.no_grad():
                saliency_map = self.model(img_tensor, center_bias)
                saliency_map = saliency_map.cpu().numpy().squeeze()
                saliency_map = cv2.resize(saliency_map, (image.size[0], image.size[1]))
                saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
                
                return saliency_map
                
        except Exception as e:
            logger.error(f"Error in saliency prediction: {str(e)}")
            raise

    def calculate_metrics(self, saliency_map: np.ndarray) -> Dict[str, float]:
        """Calculate attention metrics for the entire image"""
        try:
            saliency_uint8 = (saliency_map * 255).astype(np.uint8)
            
            # Calculate cognitive demand
            gx = cv2.Sobel(saliency_uint8, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(saliency_uint8, cv2.CV_64F, 0, 1, ksize=3)
            visual_complexity = np.mean(np.sqrt(gx**2 + gy**2)) / 255.0
            content_density = np.mean(saliency_map > np.mean(saliency_map))
            laplacian = cv2.Laplacian(saliency_uint8, cv2.CV_64F)
            information_load = np.clip(laplacian.var() / (255.0 ** 2), 0, 1)
            
            cognitive_demand = (
                visual_complexity * 0.4 +
                content_density * 0.3 +
                information_load * 0.3
            ) * 100
            
            # Calculate other metrics
            histogram = np.histogram(saliency_map, bins=10)[0]
            distribution_clarity = 1 - (np.std(histogram) / np.mean(histogram) if np.mean(histogram) > 0 else 0)
            edges = cv2.Canny(saliency_uint8, 100, 200)
            edge_clarity = np.mean(edges > 0)
            contrast = np.max(saliency_map) - np.min(saliency_map)
            
            clarity_score = (
                distribution_clarity * 0.4 +
                edge_clarity * 0.3 +
                contrast * 0.3
            ) * 100
            
            # Focus score calculation
            attention_score = np.mean(saliency_map) * 100
            height, width = saliency_map.shape
            center_y, center_x = height // 2, width // 2
            y_coords, x_coords = np.ogrid[:height, :width]
            distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_distance = np.sqrt((width/2)**2 + (height/2)**2)
            normalized_distance = distance_from_center / max_distance
            balance_score = 1 - np.mean(normalized_distance * saliency_map)
            
            sorted_values = np.sort(saliency_uint8.flatten())
            hierarchy_score = np.mean(sorted_values[-int(len(sorted_values)*0.1):]) / 255.0
            
            focus_score = (
                attention_score * 0.4 +
                balance_score * 100 * 0.3 +
                hierarchy_score * 100 * 0.3
            )
            
            # Calculate engagement score
            magnitude = np.sqrt(gx**2 + gy**2)
            pattern_complexity = np.clip(magnitude.mean() / 32.0, 0, 1) * 100
            flow = np.arctan2(gy, gx)
            motion_score = np.clip(np.std(flow) * 20, 0, 100)
            
            engagement_score = np.clip(
                attention_score * 0.25 +
                contrast * 100 * 0.20 +
                pattern_complexity * 0.25 +
                hierarchy_score * 100 * 0.20 +
                motion_score * 0.10,
                0, 100
            )
            
            return {
                'cognitive_demand': cognitive_demand,
                'clarity_score': clarity_score,
                'focus_score': focus_score,
                'engagement_score': engagement_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def generate_heatmap(self, saliency_map: np.ndarray) -> np.ndarray:
        """Convert saliency map to colored heatmap"""
        saliency_map_norm = (saliency_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(saliency_map_norm, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    def overlay_heatmap(self, image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on original image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        return cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)

# Initialize analyzer
analyzer = EnhancedAttentionAnalyzer()

# API endpoints
@app.post("/api/v1/analyze/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> JSONResponse:
    """Upload and analyze image"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if is_blank_image(np.array(image)):
            raise HTTPException(status_code=400, detail="Blank or empty image detected")
            
        analysis_id = str(uuid.uuid4())
        
        def analyze():
            saliency_map = analyzer.predict_saliency(image)
            metrics = analyzer.calculate_metrics(saliency_map)
            return {
                'image': image,
                'saliency_map': saliency_map,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(thread_pool, analyze)
        
        analysis_cache[analysis_id] = result
        
        background_tasks.add_task(clean_old_cache_entries)
        
        return JSONResponse({
            'analysis_id': analysis_id,
            'metrics': result['metrics'],
            'timestamp': result['timestamp'].isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analyze/{analysis_id}/heatmap")
async def get_heatmap(analysis_id: str, alpha: float = 0.5):
    """Get attention heatmap visualization"""
    try:
        if analysis_id not in analysis_cache:
            raise HTTPException(status_code=404, detail="Analysis not found")
            
        cached_data = analysis_cache[analysis_id]
        
        def generate_visualization():
            heatmap = analyzer.generate_heatmap(cached_data['saliency_map'])
            overlay = analyzer.overlay_heatmap(cached_data['image'], heatmap, alpha)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            Image.fromarray(overlay).save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return img_byte_arr.getvalue()
            
        loop = asyncio.get_event_loop()
        heatmap_bytes = await loop.run_in_executor(thread_pool, generate_visualization)
        
        return Response(content=heatmap_bytes, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analyze/{analysis_id}/metrics")
async def get_metrics(analysis_id: str) -> JSONResponse:
    """Get detailed metrics for a previous analysis"""
    try:
        if analysis_id not in analysis_cache:
            raise HTTPException(status_code=404, detail="Analysis not found")
            
        cached_data = analysis_cache[analysis_id]
        return JSONResponse({
            'analysis_id': analysis_id,
            'metrics': cached_data['metrics'],
            'timestamp': cached_data['timestamp'].isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/analyze/{analysis_id}")
async def delete_analysis(analysis_id: str) -> JSONResponse:
    """Delete an analysis and free up cache"""
    try:
        if analysis_id not in analysis_cache:
            raise HTTPException(status_code=404, detail="Analysis not found")
            
        del analysis_cache[analysis_id]
        return JSONResponse({
            'status': 'success',
            'message': f'Analysis {analysis_id} deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def clean_old_cache_entries():
    """Clean cache entries older than 1 hour"""
    try:
        current_time = datetime.now()
        to_delete = []
        
        for analysis_id, data in analysis_cache.items():
            if (current_time - data['timestamp']).total_seconds() > 3600:
                to_delete.append(analysis_id)
                
        for analysis_id in to_delete:
            del analysis_cache[analysis_id]
            
        logger.info(f"Cleaned {len(to_delete)} old cache entries")
        
    except Exception as e:
        logger.error(f"Error cleaning cache: {str(e)}")

@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint"""
    return JSONResponse({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_size': len(analysis_cache)
    })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=True,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=30
    )

# API documentation: http://localhost:8000/api/docs
# Alternative API docs: http://localhost:8000/api/redoc
# Health check: http://localhost:8000/health