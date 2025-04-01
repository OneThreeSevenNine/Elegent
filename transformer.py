import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "demodle"
        
        self.d_model = d_model    # ģ��ά�ȣ���512��
        self.num_heads = num_heads # ע����ͷ������8��
        self.d_k = d_model // num_heads # ÿ��ͷ��ά�ȣ���64��
        
        # �������Ա任�㣨����ƫ�ã�
        self.W_q = nn.Linear(d_model, d_model) # ��ѯ�任
        self.W_k = nn.Linear(d_model, d_model) # ���任
        self.W_v = nn.Linear(d_model, d_model) # ֵ�任
        self.W_o = nn.Linear(d_model, d_model) # ����任
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # ����ע����������Q��K�ĵ����
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Ӧ�����루����������δ����Ϣ���룩
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # ����ע����Ȩ�أ�softmax��һ����
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # ��ֵ������Ȩ���
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # ���Ա任���ָ��ͷ
        Q = self.split_heads(self.W_q(Q)) # (batch, heads, seq_len, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # ����ע����
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # �ϲ���ͷ������任
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # ��һ��ȫ����
        self.fc2 = nn.Linear(d_ff, d_model)  # �ڶ���ȫ����
        self.relu = nn.ReLU()  # �����

    def forward(self, x):
        # ǰ������ļ���
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)  # ��ʼ��λ�ñ������
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # ż��λ��ʹ�����Һ���
        pe[:, 1::2] = torch.cos(position * div_term)  # ����λ��ʹ�����Һ���
        self.register_buffer('pe', pe.unsqueeze(0))  # ע��Ϊ������
        
    def forward(self, x):
        # ��λ�ñ�����ӵ�������
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # ��ע��������
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # ǰ������
        self.norm1 = nn.LayerNorm(d_model)  # ���һ��
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout
        
    def forward(self, x, mask):
        # ��ע��������
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # �в����ӺͲ��һ��
        
        # ǰ������
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # �в����ӺͲ��һ��
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # ��ע��������
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # ����ע��������
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # ǰ������
        self.norm1 = nn.LayerNorm(d_model)  # ���һ��
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # ��ע��������
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))  # �в����ӺͲ��һ��
        
        # ����ע��������
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))  # �в����ӺͲ��һ��
        
        # ǰ������
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # �в����ӺͲ��һ��
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # ��������Ƕ��
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # ��������Ƕ��
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)  # λ�ñ���

        # �������ͽ�������
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)  # ���յ�ȫ���Ӳ�
        self.dropout = nn.Dropout(dropout)  # Dropout

    def generate_mask(self, src, tgt):
        # Դ���룺����������������������Ϊ0��
        # ��״��(batch_size, 1, 1, seq_length)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
        # Ŀ�����룺����������δ����Ϣ
        # ��״��(batch_size, 1, seq_length, 1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # ���������Ǿ������룬��ֹ����ʱ����δ����Ϣ
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask  # �ϲ���������δ����Ϣ����
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # ��������
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # ����������
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # ����������
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # �������
        output = self.fc(dec_output)
        return output
    
# ������
src_vocab_size = 5000  # Դ�ʻ���С
tgt_vocab_size = 5000  # Ŀ��ʻ���С
d_model = 512  # ģ��ά��
num_heads = 8  # ע����ͷ����
num_layers = 6  # �������ͽ���������
d_ff = 2048  # ǰ�������ڲ�ά��
max_seq_length = 100  # ������г���
dropout = 0.1  # Dropout ����

# ��ʼ��ģ��
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# �����������
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # Դ����
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # Ŀ������

# ������ʧ�������Ż���
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ������䲿�ֵ���ʧ
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# ѵ��ѭ��
transformer.train()
for epoch in range(100):
    optimizer.zero_grad()  # ����ݶȣ���ֹ�ۻ�
    
    # ����Ŀ������ʱȥ�����һ���ʣ�����Ԥ����һ���ʣ�
    output = transformer(src_data, tgt_data[:, :-1])  
    
    # ������ʧʱ��Ŀ�����дӵڶ����ʿ�ʼ����Ԥ����һ���ʣ�
    # output��״: (batch_size, seq_length-1, tgt_vocab_size)
    # Ŀ����״: (batch_size, seq_length-1)
    loss = criterion(
        output.contiguous().view(-1, tgt_vocab_size), 
        tgt_data[:, 1:].contiguous().view(-1)
    )
    
    loss.backward()        # ���򴫲�
    optimizer.step()       # ���²���
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

transformer.eval()
# ������֤����
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))
# ��������Ϊһ��Ӣ�ĺͶ�Ӧ�����ķ��루��ת��Ϊ������
# ʾ�����ݣ�
# src_data: [[3, 14, 25, ..., 0, 0], ...]  # Ӣ�ľ��ӣ�0Ϊ������
# tgt_data: [[5, 20, 36, ..., 0, 0], ...]  # ���ķ��루0Ϊ������
# ע�⣺ʵ��Ӧ��������ı����зִʡ����롢����Ԥ����
with torch.no_grad():
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")