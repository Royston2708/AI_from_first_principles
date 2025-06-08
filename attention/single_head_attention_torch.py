import torch 
import math
import numpy as np 
from typing import Optional

class SingleHeadedAttention:
    def __init__(self, Q, K, V, d_model: int =512):
        """
        Initializes an instace of SingleHeadAttention

        Args:
            Q (np.ndarry): Query Matrix
            K (np.ndarry): Key Matrix
            V (np.ndarry): Val Matrix
            d_model
        """
        self.Q = Q
        self.K = K
        self.V = V 
        self.d_model = d_model

        self.d_k = self.d_model # In multi-headed attention this would be self.d_model/self.num_heads
        self.attention_mat = None 
        self.attention_score = None 
    
    def softmax_across_row(self, X) -> torch.tensor:
        """
        We will use this method to compute the softmax along rows of the input.
        Formuls of Softmax = exp(x)/ sum(exp(x))
        """
        return torch.softmax(X, dim=1)
    

    def compute(self, mask: Optional[np.ndarray] = None) -> torch.tensor:
        """
        Compute Single Headed attention (masker or unmaked)

        
        -- 
        """
        qk_final = torch.matmul(self.Q, self.K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            qk_final = qk_final + mask 
        
        # Compute Attention Matrix 
        self.attention_mat= self.softmax_across_row(qk_final)

        # Compute Attention Score 
        self.attention_score = torch.matmul(self.attention_mat, self.V)

        return self.attention_mat, self.attention_score



if __name__ == "__main__":
    seq_len = 6
    d_model = 512 

    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)

    attention = SingleHeadedAttention(Q = Q, K= K, V=V, d_model=d_model)
    
    attention_matrix, final_val = attention.compute()
    print(attention_matrix)
        
