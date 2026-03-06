import torch
from typing import Dict, Optional
import os
from config import LAYER_ID


class VectorInjector:
    def __init__(self, vector_dir: str, model_name: str = "qwen3-8b", device: str = "cuda", model_dtype=None):
        # vector_dir already includes model_name (e.g., ./vectors/qwen3-8b)
        self.vector_dir = vector_dir
        self.device = device
        self.model_dtype = model_dtype
        self.vectors: Dict[str, torch.Tensor] = {}
        self.active_vector: Optional[torch.Tensor] = None
        self.active_role: Optional[str] = None
        self.active = False
        
        self._load_vectors()
    
    def _load_vectors(self):
        role_vector_map = {
            "solver": f"v_solver_caa_l{LAYER_ID}.pt",
            "critic": f"v_critic_caa_l{LAYER_ID}.pt"
        }
        
        for role, filename in role_vector_map.items():
            vector_path = os.path.join(self.vector_dir, filename)
            if os.path.exists(vector_path):
                vec = torch.load(vector_path, map_location=self.device)
                
                if self.model_dtype is not None:
                    vec = vec.to(dtype=self.model_dtype)
                
                vec = vec.view(1, 1, -1)
                
                self.vectors[role] = vec
                print(f"Loaded vector for {role}: {vec.shape}, dtype: {vec.dtype}")
            else:
                print(f"Warning: Vector file not found: {vector_path}")
    
    def activate(self, role: str, coeff: float = 1.0) -> bool:
        if role not in self.vectors:
            print(f"Warning: Role '{role}' not available")
            return False
        
        # 存储归一化向量（不包含系数）
        self.active_vector = self.vectors[role]
        self.active_coeff = coeff
        self.active_role = role
        self.active = True
        print(f"Activated {role} vector with coefficient {coeff}")
        return True
    
    def deactivate(self):
        self.active_vector = None
        self.active_coeff = 0.0
        self.active_role = None
        self.active = False
        print("Deactivated vector")
    
    def get_active_vector(self) -> Optional[torch.Tensor]:
        if self.active_vector is None:
            return None
        return self.active_vector * self.active_coeff
    
    def get_normalized_vector(self) -> Optional[torch.Tensor]:
        """
        获取归一化的向量（不包含系数）
        """
        return self.active_vector
    
    def get_active_coeff(self) -> float:
        """
        获取当前的注入系数
        """
        return getattr(self, 'active_coeff', 0.0)
    
    def get_active_role(self) -> Optional[str]:
        return self.active_role
    
    def is_active(self) -> bool:
        return self.active and self.active_vector is not None
    
    def get_available_roles(self) -> list:
        return list(self.vectors.keys())
