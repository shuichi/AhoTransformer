##############################
# しつこいくらいに行間をぎっしり埋めた丁寧なTransformer Decoder実装例
# decoderonly.py
# デコーダのみの Transformer で「Aho 数」判定を学習する実装例。
# encoderonly.py と同じタスクを、GPT 風のデコーダブロックだけで組む。

# Shuichi Kurabayashi
##############################

import math  # 位置エンコーディングで使う標準数学ライブラリ
import os  # チェックポイント保存用のファイル操作
import random  # 推論テストで乱数を生成する
from typing import List, Tuple  # 型ヒント用Python標準ライブラリ
# Pythonにおける型ヒント（Type Hints）は、Python 3.5で導入された機能で、
# 変数や関数の引数・戻り値に期待される型を明示的に記述できる仕組みです。

import argparse  # コマンドライン引数処理
import torch  # PyTorch 本体
import torch.nn as nn  # NN モジュールのショートカット
from torch.utils.data import Dataset, DataLoader  # データセットとローダ
from torch.utils.tensorboard import SummaryWriter # TensorBoard ログ出力用クラス。


############################
# ルール: 「Aho」かどうか
############################

def is_aho_number(n: int) -> bool:
    """
    与えられた整数が「3の倍数」または桁に「3」を含む場合に True を返す。

    Args:
        n (int): 判定したい整数。負数でもよく、内部で絶対値にして桁を調べる。

    Returns:
        bool: 条件を満たすなら True、それ以外は False。

    str(abs(int(n))) は「整数に直して符号を外し、文字列にして各桁を扱う」処理。
    """
    return (n % 3 == 0) or ("3" in str(abs(int(n))))


############################
# トークナイザ（デコーダ用）
############################

class DecoderTokenizer:
    """
    デコーダ専用トークナイザ。数字列に [SEP] と [MASK] を付け、最後の位置を予測させる。

    語彙:
        0: [PAD]
        1-10: '0'〜'9'
        11: [SEP]
        12: [MASK]
        13: [AHO]
        14: [SAFE]

    Attributes:
        max_len (int): 出力シーケンス長。数字+[SEP]+[LABEL]+PAD をここに収める。
        pad_id (int): PAD トークン ID。
        sep_id (int): 区切りトークン ID。
        mask_id (int): 予測用マスク ID。
        aho_id (int): Aho ラベルの ID（ターゲット）。
        safe_id (int): Safe ラベルの ID（ターゲット）。
        digit2id (dict[str, int]): 各数字文字を ID に写像する辞書。
    """

    def __init__(self, max_len: int = 8):
        self.max_len = max_len
        self.pad_id = 0
        self.sep_id = 11
        self.mask_id = 12  # 予測させる位置のマスクトークン
        self.aho_id = 13
        self.safe_id = 14
        self.digit2id = {str(d): d + 1 for d in range(10)}  # [PAD]=0 を避け 1 始まり

    @property
    def vocab_size(self) -> int:
        return 15  # 0〜14

    def encode(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        整数 n を 1 サンプル分のトークン列に変換する。

        Args:
            n (int): 変換する整数。

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                input_ids: 数字列 + [SEP] + [MASK] + PAD（shape: (max_len,)）
                attention_mask: 1=有効, 0=PAD（shape: (max_len,)）
                label_id: 正解ラベル [AHO]/[SAFE] のトークン ID（shape: ()）
        """
        s = str(abs(int(n)))
        digit_ids = [self.digit2id[ch] for ch in s]  # 各桁を ID に

        # 正解ラベル ID を決定（入力には入れずターゲットとして保持）
        if is_aho_number(n):
            label_id = self.aho_id
        else:
            label_id = self.safe_id

        # 入力系列: digits + [SEP] + [MASK]
        tokens = digit_ids + [self.sep_id] + [self.mask_id]

        # 長すぎる場合は末尾だけ残す（今回は max_len を十分大きく取る前提）
        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]  # 後ろだけ残す

        attention_mask = [1] * len(tokens)

        # PAD で右側パディング
        while len(tokens) < self.max_len:
            tokens.append(self.pad_id)
            attention_mask.append(0)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)
        return input_ids, attention_mask, label_id_tensor

    def batch_encode(
        self, numbers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        複数の整数をまとめてトークン化し、バッチテンソルを返す。

        Args:
            numbers (List[int]): 変換する整数のリスト。

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 各サンプルの input_ids, attention_mask, label_id を
            先頭軸でまとめたテンソル。
        """
        encoded = [self.encode(n) for n in numbers]
        input_ids = torch.stack([e[0] for e in encoded], dim=0)
        attention_mask = torch.stack([e[1] for e in encoded], dim=0)
        label_ids = torch.stack([e[2] for e in encoded], dim=0)
        return input_ids, attention_mask, label_ids


############################
# 位置エンコーディング（batch_first）
############################

class PositionalEncoding(nn.Module):
    """
    batch_first=True (B, S, E) 前提の位置エンコーディング。

    Attributes:
        pe (Tensor): 事前計算済みの正弦波位置ベクトル（形: (1, max_len, d_model)）。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置エンコーディングを入力に加算する。

        Args:
            x (Tensor): 形状 (B, S, E) の埋め込みベクトル。

        Returns:
            Tensor: 同形状で位置情報を足したテンソル。
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len] # pyright: ignore[reportIndexIssue]


############################
# デコーダブロック（自己注意のみ）
############################

class DecoderBlock(nn.Module):
    """
    GPT 風の自己注意 + FFN ブロック。
    Encoder も Cross-Attention も持たない「デコーダ単体」。
    
    Args:
        d_model: 隠れ次元数。
        nhead: マルチヘッド数。
        dim_feedforward: FFN の中間次元。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        #              Self-Attention: 第1段階
        #     （X から Q / K / V を生成し、ヘッドに分割）
        #     入力埋め込み列 X
        #     形: (B, S, d_model)
        #
        #                     X
        #       （Self-Attention では query = key = value = X）
        #                     │
        #                     │  線形変換（全結合）
        #                     │  W_Q, W_K, W_V
        #         ┌───────────┼───────────────┬───────────────┐
        #         │           │               │               │
        #         ▼           ▼               ▼               ▼
        #    Q_all_heads   K_all_heads    V_all_heads
        # 形: (B, S, d_model) (B, S, d_model) (B, S, d_model)
        #         │               │               │
        #         │   ヘッド方向に分割（d_model = h * d_k）＋ reshape / transpose
        #         │
        #         ├────▶ Q_heads: (B, h, S, d_k)
        #         ├────▶ K_heads: (B, h, S, d_k)
        #         └────▶ V_heads: (B, h, S, d_k)        
        #
        #
        #
        #                     Self-Attention: 第2段階
        #       （各ヘッドでスケールド・ドット積アテンション → ヘッド結合）
        #
        # Q_heads: (B, h, S, d_k)
        # K_heads: (B, h, S, d_k)
        # V_heads: (B, h, S, d_k)
        #                     │
        #                     │  内積 + スケーリング
        #                     ▼
        #     score = Q_heads @ K_heads^T / sqrt(d_k)
        #     形: (B, h, S, S)
        #                     │
        #                     │  softmax（キー方向 S で正規化）
        #                     ▼
        #     attn_weights = softmax(score, dim = -1)
        #     形: (B, h, S, S)
        #                     │
        #                     │  V_heads との重み付き和
        #                     ▼
        #     head_outputs = attn_weights @ V_heads
        #     形: (B, h, S, d_k)
        #                     │
        #                     │  ヘッド方向の結合（concat）
        #                     ▼
        #     concat = reshape(head_outputs, (B, S, h * d_k))
        #     形: (B, S, d_model)
        #                     │
        #                     │  出力用線形変換 W_O, b_O
        #                     ▼
        #     out = concat @ W_O^T + b_O
        #     形: (B, S, d_model)        
        
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
        # Self-Attention（DecoderBlock.forward 内の self.self_attn 付近の行）では、
        # 因果マスクによって各トークンが自分より前のトークンだけにアテンションできるようにしつつ、
        # key_padding_mask によってパディング位置は無視します。
        # これにより系列内の位置をまたいで情報が混ざるため、
        # モデルはプレフィックス全体（digits + [SEP] + [MASK]）を利用して最終的なラベルトークンを判断できます。
        # Feed-Forward Network（linear1 → activation → dropout → linear2）は、
        # アテンションの出力に対して位置ごとの非線形変換を適用します。
        # 一度 dim_feedforward 次元まで拡張し、そこから d_model に再投影することで、
        # 非線形性を加えつつ、アテンションでコンテキストが混ざった後の各トークンの特徴量を細かく整形します。
        # より端的には、FFNは「Self-Attentionで収集した情報を、その位置ごとに“意味のある特徴”に変換するための
        # 非線形な変換装置」です。まず Self-Attention で「どのトークンがどのトークンからどれくらい情報をもらうか」が決まり
        # （つまり、「Aho」判定という観点からの数字と数字の関係が決まり）、
        # 各位置に「コンテキストが混ざったベクトル」ができます。
        # ただ、この段階では「足し合わせ+線形変換+softmax」程度の構造なので、表現力としては限定的です。
        # そこで FFN が、各位置のベクトルに対して独立に、しかし非線形な多層パーセプトロンとして働き、
        # 「このパターンが来たらこういう特徴を強調する／抑える」といった形で特徴を再構成します。
        # 学習という観点では、たとえば「この桁パターンのときは [AHO] にしたい」
        # 「この文脈のときは動詞の活用をこうする」といった、入力と出力の間の複雑な対応関係を、
        # FFN が各位置ごとの非線形写像としてモデル化していると言えます。
        # Self-Attention が「どの情報を集めるか」を決める“ルーティング”の役割だとすると、
        # FFN は「集めてきた情報からどういう決定境界・特徴抽出を学習するか」を担っている、
        # と考えると分かりやすいと思います。
        # また、実際の Transformer ではパラメータの大半が FFN に集中しており、
        # ここがモデルの表現力を大きく左右しています。
        # これまでの実験でも、Self-Attentionを残してFFNを削減すると性能が大きく低下することが知られています。
        # つまり、
        # 「コンテキストを識別・合成するのがSelf-Attention」で、
        # 「コンテキスト依存のベクトルをタスクに役立つ形に整形・調整するのがFFN」 です。

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): 入力特徴 (B, S, E)。
            attn_mask (Tensor): 未来を隠す causal mask。shape (S, S)。
            key_padding_mask (Tensor): PAD 位置を True にしたマスク。shape (B, S)。

        Returns:
            Tensor: ブロック通過後の特徴 (B, S, E)。
        """
        # Self-Attention (causal + padding mask)
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # 自己注意の計算です。
        # self.self_attn(x, x, x, ...) はクエリ・キー・バリューすべてに同じテンソル x（形: B, S, E）を使う「自己注意」。
        # attn_mask は (S, S) の因果マスクで、未来トークンを見ないよう上三角をマスクします。
        # key_padding_mask は (B, S) で PAD 位置を True にし、その位置を無視させます。
        # need_weights=False で注意重み行列を返さず、出力だけ受け取ります。
        # 結果の attn_output（文脈を混ぜた特徴）を元の x に残差接続して、後段の LayerNorm へ渡しています。
        
        # 少し長くなりますが、自己注意機構を詳しく説明します。
        # まず「クエリ・キー・バリューすべてに同じテンソル x を使う」と書きましたが、自己注意では、各トークンに対して
        # 「何を探すか」を表すベクトル（クエリ）、
        # 「自分はどういう情報を提供できるか」を表すベクトル（キー）、
        # 「実際に相手に渡す中身」を表すベクトル（バリュー）、
        # この三つを、それぞれ別々の「学習した変換ルール（バイアス）」を経由して計算します。
        # つまり、変換ルールが学習過程で動的に変化するので、たとえ元の入力ベクトル x が同じデータから来ていても、
        # QKVの全てが違う、という状態になります。
        # また、「位置埋め込み」の影響も重要です。文章で言うと「同じ単語でも、文頭にあるか文末にあるかで意味合いが変わる」ので、
        # モデルは各位置に違う“位置のラベル”を足してから自己注意に渡します。
        # したがって、そもそも列の中の各トークンは、内容も位置も含めた「その場所固有のベクトル」になっています。
        # 同じ x という言い方をしていても、実際には“位置 i の x”と“位置 j の x”は別物です。
        # これにより、数字の桁数をデコーダーが学習できるようになります。
        # 同じ x を起源としても、最初にランダムに与えられた「クエリ用の変換バイアス」「キー用の変換バイアス」「バリュー用の変換バイアス」が、
        # 損失を小さくする方向にそれぞれ別々に調整され続けるため、「クエリ側は“何を探しているか”を強く表現するように」
        # 「キー側は“どういうときに参照されるべきか”が分かるように」
        # 「バリュー側は“参照されたときに渡すべき情報”をうまく持てるように」というように、自然と非対称な役割が形成されます。
        # 最初は何もできないけれど、誤差逆伝播を繰り返すうちに、「このタスクでは、こういうときにここのトークンを見に行くと正解になりやすい」
        # というパターンが埋め込まれていきます。
        # こうして、各位置のトークンが、自分のクエリを持って、列の中のキーたちを見比べて、
        # このトークンから情報を受け取る、このトークンは無視する、と重みづけすることで、
        # “どこからコンテキストを集めるか”が決まります。
        # そのうえで、選ばれたトークンたちから「どんな特徴を受け取るか」はバリューが決め、
        # その結果をあと段の Feed-Forward Network がさらに非線形に加工します。
                
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Position-wise FFN
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


def generate_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    長さ sz の系列用に、未来を見ないようにする causal mask を作る。

    True / -inf が「見えない」位置になるように MultiheadAttention に渡す。
    PyTorch 2.x では bool マスクも float マスクも受け付ける。
    """
    # shape: (sz, sz)
    # 上三角（対角の1つ上から）を True にして「見えない」位置とする
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
    return mask


############################
# デコーダ専用 Transformer モデル
############################

class AhoDecoderTransformer(nn.Module):
    """
    デコーダ単体 Transformer。
    入力: 数字列 + [SEP] + [AHO/SAFE] + PAD
    出力: 各位置の語彙分布（特に最後の [AHO/SAFE] 位置を使う）

    Args:
        vocab_size: 語彙サイズ（14 固定だが外から渡せる形にしている）。
        max_len: シーケンス長。位置エンコーディングやマスク生成で使う。
        d_model: 隠れ次元数。
        nhead: マルチヘッド数。
        num_layers: デコーダブロックの段数。
        dim_feedforward: FFN の中間次元。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids (Tensor): 形状 (B, S) のトークン ID。
            attention_mask (Tensor): 形状 (B, S) のマスク。1=有効, 0=PAD。

        Returns:
            Tensor: 形状 (B, S, vocab_size) のロジット。
        """
        device = input_ids.device
        B, S = input_ids.size()

        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # (B, S, E)
        x = self.pos_encoding(x)

        # causal mask: (S, S) でバッチ共通
        attn_mask = generate_subsequent_mask(S, device)

        # key_padding_mask: True が PAD
        key_padding_mask = attention_mask == 0  # (B, S) bool

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        logits = self.lm_head(x)  # (B, S, V)
        return logits


############################
# Dataset と学習ループ
############################

class AhoDecoderDataset(Dataset):
    """
    numbers の各整数を「数字列 + [SEP] + [LABEL]」に変換する Dataset。

    入力: トークン ID 列と attention mask
    ラベル: [AHO] または [SAFE] の ID
    """

    def __init__(self, numbers: List[int], tokenizer: DecoderTokenizer):
        self.numbers = numbers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx: int):
        n = self.numbers[idx]
        input_ids, attention_mask, label_id = self.tokenizer.encode(n)
        return input_ids, attention_mask, label_id, n


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """1 epoch 学習し、平均損失を返す。"""
    model.train()
    total_loss = 0.0
    total_count = 0

    criterion = nn.CrossEntropyLoss()

    for input_ids, attention_mask, label_ids, _ in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)  # (B, S, V)

        # ラベル位置は「最後の有効トークン」（mask=1 の最後）
        # pos: (B,)
        label_positions = attention_mask.sum(dim=1) - 1
        B, S, V = logits.size()

        # (B, V) に取り出す
        idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)
        logits_label = logits.gather(1, idx).squeeze(1)  # (B, V)

        loss = criterion(logits_label, label_ids)
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    検証用ループ。平均損失と精度を返す。
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0

    criterion = nn.CrossEntropyLoss()

    for input_ids, attention_mask, label_ids, _ in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)

        logits = model(input_ids, attention_mask)  # (B, S, V)

        label_positions = attention_mask.sum(dim=1) - 1
        B, S, V = logits.size()
        idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)
        logits_label = logits.gather(1, idx).squeeze(1)  # (B, V)

        loss = criterion(logits_label, label_ids)

        preds = logits_label.argmax(dim=-1)
        correct = (preds == label_ids).sum().item()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        total_correct += correct

    return total_loss / total_count, total_correct / total_count


############################
# 推論: 「Aho」かどうかを表示
############################

@torch.no_grad()
def aho_infer(
    model: AhoDecoderTransformer,
    tokenizer: DecoderTokenizer,
    numbers: List[int],
    device: torch.device,
):
    """
    与えられた整数リストを推論し、結果をコンソールに表示する。
    """
    model.eval()
    input_ids, attention_mask, label_ids = tokenizer.batch_encode(numbers)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(input_ids, attention_mask)
    label_positions = attention_mask.sum(dim=1) - 1
    B, S, V = logits.size()
    idx = label_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, V)
    logits_label = logits.gather(1, idx).squeeze(1)  # (B, V)

    probs = logits_label.softmax(dim=-1)
    preds = probs.argmax(dim=-1).cpu().tolist()

    total = len(numbers)
    correct = 0
    for i, n in enumerate(numbers):
        pred_id = preds[i]
        p_aho = probs[i, tokenizer.aho_id].item()
        p_safe = probs[i, tokenizer.safe_id].item()
        if pred_id == tokenizer.aho_id:
            tag = "Aho"
        elif pred_id == tokenizer.safe_id:
            tag = "Safe"
        else:
            tag = f"Other({pred_id})"
        if is_aho_number(n) == (pred_id == tokenizer.aho_id):
            correct += 1
        print(
            f"{n:5d} -> {tag} "
            f"(p_AHO={p_aho:.3f}, p_SAFE={p_safe:.3f}, rule={is_aho_number(n)})"
        )
    acc = correct / total if total > 0 else 0.0
    print(f"\n正解率: {acc*100:.2f}% ({correct}/{total})")


############################
# メイン
############################

def save_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """モデルとオプティマイザの状態を dict にまとめて保存する。"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="学習済みチェックポイントへのパス（指定時は推論のみ実行）",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="対話モードで数字を入力して判定（チェックポイント指定時のみ有効）",
    )
    args = parser.parse_args()

    if torch.backends.mps.is_available():          # Mac (M1/M2/M3 など) のGPU(MPS)
        device = torch.device("mps")
    elif torch.cuda.is_available():                # NVIDIA GPU (Linux / Windows 等)
        device = torch.device("cuda")
    else:                                          # どちらも無ければCPU
        device = torch.device("cpu")
    print("device:", device)

    # 5桁まで想定（数字桁最大5 + [SEP] + [LABEL] = 7）なので max_len=7 くらいで足りる
    tokenizer = DecoderTokenizer(max_len=7)

    # 学習用・検証用の範囲
    train_numbers = list(range(1, 40001))
    val_numbers = list(range(40001, 50001))

    train_dataset = AhoDecoderDataset(train_numbers, tokenizer)
    val_dataset = AhoDecoderDataset(val_numbers, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = AhoDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        d_model=64,
        nhead=4,
        num_layers=2,       # デコーダブロック数（1にすれば「1ブロックだけ」）
        dim_feedforward=256,
        dropout=0.05,
    ).to(device)

    # チェックポイント指定時はロードして学習をスキップ
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint} "
              f"(epoch={checkpoint.get('epoch', 'N/A')})")
        model.eval()

        if args.interactive:
            print("\n=== 対話モード ===")
            print("数字を入力してください。空行または Ctrl+C / Ctrl+D で終了します。")
            while True:
                try:
                    raw = input("n> ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n終了します。")
                    break

                if raw == "":
                    print("終了します。")
                    break

                try:
                    num = int(raw)
                except ValueError:
                    print("整数を入力してください。")
                    continue

                # 1 サンプルだけをバッチ化して推論する
                input_ids, attention_mask, _ = tokenizer.batch_encode([num])
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    logits = model(input_ids, attention_mask)
                    label_pos = attention_mask.sum(dim=1) - 1  # 最後の有効トークン
                    _, _, vocab = logits.size()
                    idx = label_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, vocab)
                    logits_label = logits.gather(1, idx).squeeze(1)  # (1, V)
                    probs = logits_label.softmax(dim=-1)

                pred_id = probs.argmax(dim=-1).item()
                p_aho = probs[0, tokenizer.aho_id].item()
                p_safe = probs[0, tokenizer.safe_id].item()
                pred_is_aho = pred_id == tokenizer.aho_id
                rule = is_aho_number(num)
                tag = "Aho" if pred_is_aho else "Safe"
                correct = " / 正解" if pred_is_aho == rule else ""

                print(
                    f"モデル判定: {num} -> {tag} "
                    f"(p_AHO={p_aho:.3f}, p_SAFE={p_safe:.3f}) "
                    f"/ ルール: {rule}{correct}"
                )
        else:
            print("\n=== デコーダ Transformer によるAho判定 ===")
            n = 1000
            test_numbers = [random.randint(10000, 99999) for _ in range(n)]
            aho_infer(model, tokenizer, test_numbers, device)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        num_epochs = 100
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter()
        ckpt_dir = "checkpoints"

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc*100:.2f}%"
            )
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            if epoch % 10 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(ckpt_path, model, optimizer, epoch)
                print(f"Saved checkpoint: {ckpt_path}")
