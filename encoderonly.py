##############################
# しつこいくらいに行間をぎっしり埋めた丁寧なTransformer Encoder実装例
# encoderonly.py
# エンコーダのみの Transformer を使った
# 「3の倍数 or 3を含むか」判定モデルの実装例

# Shuichi Kurabayashi
##############################

import math # 位置エンコーディングに使う、Python標準数学ライブラリ
import os # ファイル操作に使う、Python標準ライブラリ
import random # テスト用の乱数生成に使う、Python標準ライブラリ
from typing import List, Tuple # 型ヒント用、Python標準ライブラリ

############################
# ここまでは標準ライブラリをインポートするだけの処理。
# 一般的なPythonコードでは、数学処理とファイル操作と型ヒントはよく使うので、
############################

import argparse # コマンドライン引数処理に使う、サードパーティライブラリ

import torch # PyTorch本体。
import torch.nn as nn # ニューラルネットワーク用のレイヤー/損失などのクラス。
import torch.nn.functional as F # レイヤーの関数版（活性化関数や畳み込みなどを直接呼ぶため）。
from torch.utils.data import Dataset, DataLoader # データセット定義用の基底クラスと、バッチ化・シャッフルなどを行うデータローダ。

from torch.utils.tensorboard import SummaryWriter # TensorBoard ログ出力用クラス。

############################
# ルール: 「Aho」かどうかを判定する関数。
# 今回のTransformerはこのルールを「コードを読まずにデータから」学習する。
############################

def is_aho_number(n: int) -> bool:
    """
    与えられた整数が「3の倍数」または桁に「3」を含む場合に True を返す。

    Args:
        n (int): 判定対象の整数。符号付きでもよく、内部で絶対値にして桁を調べる。

    Returns:
        bool: 条件を満たせば True、それ以外は False。
        
    str(abs(int(n))) は n を整数に直し、その絶対値を取り、文字列に変換する処理です。
    int(n): n を整数型にキャスト（小数や整数風の文字列にも対応）。
    abs(...): 符号を外し正の値にする（負の数でも桁を調べやすくする）。
    str(...): 数値を文字列化し、各桁を文字として扱えるようにする（ここでは "3" を含むか判定するため）。    
    """
    return (n % 3 == 0) or ("3" in str(abs(int(n))))



############################
# トークナイザ

# Transformerにテキスト列を学習させるとき、そのままの文字列では扱えないので、
# まずトークンという離散的な ID の列に変換します。
# ここで問題になるのが、「語彙（トークンの種類）をどう設計するか」です。
# 語彙を「単語」だけで構成すると、次のような問題が出ます。
# 文字レベルだけにすると、語彙数は少なくて済む一方で、1 文が非常に長いトークン列になってしまい、学習・推論が非効率になります。
# 単語レベルにすると、頻度の低い単語が山ほど出てきて「知らない単語（OOV）」問題が深刻になるうえ、語彙サイズも巨大化します。
# そこで開発されたトークナイザが、BPE（Byte Pair Encoding）です。
# BPEの基本は、「頻度の高い隣り合うペアを、1 つの新しい記号として置き換え続ける」ことです。
# 「よく出る部分文字列は 1 トークンにまとめて短くしつつ、めったに出ない単語でも細かく分割すれば必ず表現できる」というバランスを取ろうとします。
# ここでは、BPEのような汎用のトークナイザは使わず、単純に「各桁を1トークン」とするトークナイザを実装します。
############################

class DigitTokenizer:
    """
    整数を桁ごとのトークンID列に変換し、必要に応じて [CLS]/[PAD] を付与するトークナイザ。

    語彙:
        0: [PAD]
        1-10: '0'〜'9'
        11: [CLS]

    Attributes:
        max_len (int): 生成するシーケンス長。
        pad_token_id (int): パディングトークンID。
        cls_token_id (int): 先頭に付与する特殊トークンID。
        digit2id (dict[str, int]): 文字数字からIDへの写像。
    """

    def __init__(self, max_len: int = 6):
        """
        トークナイザを初期化する。

        Args:
            max_len (int, optional): 生成シーケンス長。[CLS] を含めた長さ。
                例えば max_len=6 のとき、[CLS]+5桁 までを扱う。
        """
        self.max_len = max_len
        self.pad_token_id = 0
        self.cls_token_id = 11
        self.digit2id = {str(d): d + 1 for d in range(10)}
        # 0〜9 を文字としてキーにし、値を 1〜10 に割り当てる辞書を生成しています。
        # range(10) で 0..9 を列挙
        # str(d) で各桁を文字キーに
        # d + 1 で [PAD]=0 を避け、数字のトークンIDを 1 から始める番号にする

    @property
    def vocab_size(self) -> int:
        return 12  # 0〜11

    def encode_number(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        単一の整数をトークンID列と attention mask に変換する。

        Args:
            n (int): 変換する整数。

        Returns:
            Tuple[Tensor, Tensor]:
                input_ids (LongTensor): 形状 (max_len,) のトークンID列。
                attention_mask (LongTensor): 形状 (max_len,) のマスク。1=有効、0=PAD。
        """
        s = str(abs(int(n)))
        # n を整数に直して符号を外したうえで文字列にしています。
        # int(n): n を整数にキャスト
        # abs(...): マイナスでも桁を正として扱うため絶対値を取る
        # str(...): 桁を文字として並べた文字列に変換（後続の桁ごとの処理用）
        
        digit_ids = [self.digit2id[ch] for ch in s]
        # s にある各桁文字を digit2id 辞書でトークンIDへ変換し、リストに並べています。
        # s が "203" なら ['2','0','3'] をそれぞれ 3,1,4 のIDに置き換えて [3, 1, 4] のようなリストを作る処理です。

        tokens = [self.cls_token_id] + digit_ids
        # 先頭に特殊トークン [CLS] のIDを付けて、続く桁ID列 digit_ids と連結し、最終的なトークン列を作っています。
        # [self.cls_token_id] と digit_ids はどちらもリストなので、この + はリスト同士の連結（結合）を行い、新しいリストを返します。
        # 数字の加算や要素同士の演算ではありません。

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        # 生成したトークン列が設定した最大長 max_len を超えた場合に先頭から max_len 要素だけ残して切り詰めています。
        # これで [CLS] 以降が長すぎてもシーケンス長を一定に保ちます。

        attention_mask = [1] * len(tokens)
        # 現在のトークン列の長さぶんだけ 1 を並べたリストを作り、
        # 各トークンが有効（PAD ではない）であることを示すマスクとして初期化しています。
        # [1] * len(tokens) の * はリストをその回数だけ繰り返して新しいリストを作る演算です

        while len(tokens) < self.max_len:
            tokens.append(self.pad_token_id)
            attention_mask.append(0)
        # トークン列が max_len に達するまで [PAD] のIDを末尾に追加し、
        # 同じ位置の attention_mask には「無効」を示す 0 を積み増すループです。

        input_ids = torch.tensor(tokens, dtype=torch.long)
        # tokens のリストを PyTorch のテンソル（torch.long 型の1次元 LongTensor）に変換しています。
        # モデル入力として扱える数値配列にする処理です。
        
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # attention_mask のリストを PyTorch のテンソル（torch.long 型の1次元 LongTensor）に変換しています。
        # モデルがマスクを数値配列として扱えるようにする処理です。
        
        return input_ids, attention_mask

    def batch_encode_numbers(
        self, numbers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        複数の整数をまとめてトークン化し、トークンID列とattention maskの
        PyTorchで扱えるバッチ形式のテンソルとして変換する関数です。
        具体的には、各整数を encode_number で [CLS]+桁列+[PAD] のトークンIDとマスクに変換し、
        それらを先頭軸で積んで形状 (バッチサイズ, max_len) の input_ids と attention_mask を返します。

        Args:
            numbers (List[int]): 変換する整数のリスト。

        Returns:
            Tuple[Tensor, Tensor]:
                input_ids (LongTensor): 形状 (B, max_len) のトークンID列。
                attention_mask (LongTensor): 形状 (B, max_len) のマスク。1=有効、0=PAD。
        """
        encoded = [self.encode_number(n) for n in numbers]
        # numbers に含まれる各整数ごとに encode_number を呼び、
        # 結果のタプル (input_ids, attention_mask) をリストに集めています。
        # リスト内包表記で一括変換している部分です。
        
        input_ids = torch.stack([e[0] for e in encoded], dim=0)
        # encoded に格納された各サンプルの input_ids を取り出し、
        # dim=0 でスタックしてバッチ次元を持つ1本のテンソル (batch_size, max_len) を作っています。
        # リスト内包表記で最初の要素だけ抜き出し、torch.stack で縦方向に積んでいます。
        # 「スタックしてバッチ次元を持つ1本のテンソル (batch_size, max_len) を作る」とは、
        # 複数のサンプルを縦に積んで「まとめた配列」を作っている、ということです。
        # 各サンプルの input_ids は長さ max_len の1次元テンソル。
        # それらを dim=0（先頭方向）に積むと、先頭に「何個のサンプルか」を表す軸が増え、
        # 形状が (サンプル数, max_len) なります。
        # こうして、1個ずつの配列を「バッチ」として1つの2次元テンソルにまとめています。

        attention_mask = torch.stack([e[1] for e in encoded], dim=0)
        # encoded に入っている各リストの attention_mask を取り出し、
        # 先頭軸（dim=0）に積んでバッチ形状 (バッチサイズ, max_len) の1本のテンソルにまとめています。
        
        return input_ids, attention_mask


############################
# 位置エンコーディング
############################

class PositionalEncoding(nn.Module):
    """
    Transformer で用いる正弦波位置エンコーディングを計算し、入力埋め込みに加算するモジュール。

    Attributes:
        pe (Tensor): 形状 (max_len, 1, d_model) の事前計算済み位置エンコーディング。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        """
        位置エンコーディングを事前計算し、バッファに登録する。

        Args:
            d_model (int): 埋め込み次元数。
            max_len (int, optional): 事前計算する最大シーケンス長。
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルに位置エンコーディングを加算する。

        Args:
            x (Tensor): 形状 (seq_len, batch_size, d_model) の埋め込みテンソル。

        Returns:
            Tensor: 位置エンコーディングを加算したテンソル。形状は入力と同じ。
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len] # pyright: ignore[reportIndexIssue]


############################
# Transformer 本体
############################

class AhoTransformerClassifier(nn.Module):
    """
    「3の倍数 or 3を含むか」を判定する二値分類 Transformer モデル。

    Attributes:
        embedding (nn.Embedding): トークンIDを埋め込みに変換する層。
        pos_encoder (PositionalEncoding): 位置エンコーディングを付与するモジュール。
        transformer_encoder (nn.TransformerEncoder): エンコーダブロック本体。
        classifier (nn.Linear): プーリング後の特徴を1次元のロジットに射影する線形層。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 32,
    ):
        """
        モデルを初期化する。

        Args:
            vocab_size (int): 語彙サイズ。
            d_model (int, optional): 埋め込みおよびモデルの隠れ次元。
            nhead (int, optional): マルチヘッドアテンションのヘッド数。
            num_layers (int, optional): エンコーダ層のスタック数。
            dim_feedforward (int, optional): フィードフォワード層の次元。
            dropout (float, optional): ドロップアウト率。
            max_len (int, optional): 位置エンコーディングの最大長。
        """
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # 語彙サイズ vocab_size の埋め込み層を作り、各トークンIDを次元 d_model のベクトルに変換します。
        # padding_idx=0 により ID 0 の埋め込みは学習せず 0 ベクトルとして扱われ、損失計算にも影響しないようになります。
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        # 埋め込み後の特徴に正弦波の位置エンコーディングを足すためのモジュールを初期化しています。
        # d_model 次元の特徴に対応し、扱うシーケンス長の上限を max_len として事前計算します。

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        # Transformer エンコーダの1層ぶんのブロックを作成しています。
        # 隠れ次元は d_model、マルチヘッド数は nhead、中間のフィードフォワード層の次元は dim_feedforward、
        # ドロップアウト率は dropout。
        # batch_first=False なので入力・出力の形は (seq_len, batch_size, d_model) を前提に動作します。
        # batch_first は PyTorch の RNN/Transformer などで入力テンソルの次元順を指定するフラグです。
        # batch_first=True: 形は (batch_size, seq_len, feature_dim) が前提。
        # batch_first=False（デフォルトが多い）: 形は (seq_len, batch_size, feature_dim) が前提。
        # これに合わせて入力を転置したり、出力を元に戻したりします。
        # PyTorch の nn.TransformerEncoder は batch_first=False の場合、
        # 入力と出力の形を「(シーケンス長, バッチサイズ, 特徴次元)」で受け渡します。
        # つまり先頭の軸がシーケンス長になる前提なので、
        # 呼び出し前に (B, S, E) → (S, B, E) に転置し、
        # 処理後は逆に (S, B, E) → (B, S, E) に戻す必要があります。
        # (B, S, E) はテンソルの形を略記した表記で、
        # B: バッチサイズ（何サンプル分まとめているか）
        # S: シーケンス長（トークン数）
        # E: 埋め込みや特徴ベクトルの次元数
        # という3軸を持つことを示しています。
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # 先ほど作ったエンコーダ層 encoder_layer を num_layers 回積み重ねた TransformerEncoder 本体を組み立てています。
        # これが入力シーケンスを複数層通して特徴抽出する部分です。
        
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        トークンID列を受け取り、Aho判定のロジットを返す。

        Args:
            input_ids (Tensor): 形状 (batch_size, seq_len) のトークンID。
            attention_mask (Tensor): 形状 (batch_size, seq_len) のマスク。1=有効、0=PAD。

        Returns:
            Tensor: 形状 (batch_size,) のスカラー出力（ロジット）。
        """
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  
        # トークンIDを埋め込みベクトルに変換し、埋め込みの分散をそろえるためにベクトル全体を sqrt(d_model) で
        # スケーリングしています。出力は形状 (バッチ, シーケンス長, 埋め込み次元) のテンソルです。
        
        x = x.transpose(0, 1)  
        # 埋め込みのテンソルの軸を入れ替え、形を (seq_len, batch_size, embedding_dim) にしています。
        # TransformerEncoder が batch_first=False 前提なので、先頭をシーケンス長に合わせるための転置です。
        
        x = self.pos_encoder(x)
        # シーケンス位置に応じた正弦波の位置エンコーディングを埋め込みテンソルに加算しています。
        # これでトークンの並び順情報をモデルに渡します。
        
        src_key_padding_mask = (attention_mask == 0)  # (B, S) bool, True=無視
        # attention_mask で 0 の位置を True にしたブールマスクを作り、
        # Transformer へ「ここは PAD なので無視して」と伝えるための src_key_padding_mask を
        # 用意しています（形状は (batch, seq_len) ）。
        # (attention_mask == 0)は、PyTorchの演算子で、attention_mask の各要素を 0 と比較し、
        # 0 の位置だけ True、それ以外は False のブールテンソルを作っています。
        # 形状は元のマスクと同じ (batch_size, seq_len) のままです。
        
        encoded = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        ) 
        # 位置エンコーディングを付与した入力 x を Transformer エンコーダに通し、
        # src_key_padding_mask で PAD 位置を無視しながら自己注意とフィードフォワードを複数層適用して、
        # 形状 (seq_len, batch_size, d_model) のエンコード結果 encoded を得ています。

        encoded = encoded.transpose(0, 1) 
        # エンコーダ出力の軸を入れ替え、(seq_len, batch_size, d_model) から (batch_size, seq_len, d_model) に
        # 戻しています。バッチ先頭の形にして後段の処理で扱いやすくするためです。

        mask = attention_mask.unsqueeze(-1)  
        # attention_mask (B, S)の末尾に次元を1つ増やして (B, S, 1) にしています。
        # 各トークン位置のマスクを特徴次元にブロードキャストできる形にするための unsqueeze です。
        # unsqueeze はテンソルに長さ1の次元を挿入する操作です。
        # 例えば x.shape == (B, S) で x.unsqueeze(-1) とすると末尾に軸を足して (B, S, 1) になります。
        # 特徴次元などへブロードキャストしやすくする用途で使います。
        # ブロードキャストとは、配列（テンソル）の次元や長さが合わないとき、
        # 自動的に長さ1の軸を伸ばして形を揃え、要素を繰り返して演算できるようにする仕組みです。
        # 例えば (B, S, 1) と (B, S, E) を掛け算すると、
        # 前者の末尾の長さ1が E に広がって各特徴次元に同じマスクを適用できます。
        # (B, S) のままでは右端の次元が E と合わずブロードキャストできませんが、
        # 末尾に 1 を足して (B, S, 1) にすると、1 が E に拡がって (B, S, E) と要素ごとの掛け算ができるようになります。
        
        masked_encoded = encoded * mask
        # エンコーダ出力 encoded にマスクを掛けて、PAD 位置の特徴を0にしています。
        # 形状は (B, S, E) のままです。
        
        lengths = mask.sum(dim=1).clamp(min=1) 
        # 各サンプルの有効なトークン数（PAD でないトークン数）を数えています。
        # mask は (B, S, 1) なので、dim=1 で合計すると (B, 1) の各サンプルの有効トークン数が得られます。
        # clamp(min=1) で最小値を1にして、ゼロ除算を防いでいます。
        
        pooled = masked_encoded.sum(dim=1) / lengths 
        # 各サンプルの有効トークン位置の特徴を足し合わせて平均を取り、
        # 形状 (B, E) のプーリング特徴 pooled を得ています。
        # これがシーケンス全体を表す固定長の特徴ベクトルになります。

        logits = self.classifier(pooled).squeeze(-1) 
        # プーリング特徴 pooled を線形層に通して形状 (B, 1) のロジットを得てから、
        # squeeze(-1) で末尾の次元を削除し、形状 (B,) の1次元テンソルにしています。
        
        return logits


############################
# 合成データセット
############################

class AhoNumberDataset(Dataset):
    """
    numbers リストの各整数に対して、
    入力: 桁トークン列
    ラベル: Ahoなら 1.0, それ以外 0.0
    を返す Dataset。
    """

    def __init__(self, numbers: List[int], tokenizer: DigitTokenizer):
        self.numbers = numbers
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.numbers)

    def __getitem__(self, idx: int):
        n = self.numbers[idx]
        input_ids, attention_mask = self.tokenizer.encode_number(n)
        label = 1.0 if is_aho_number(n) else 0.0
        label = torch.tensor(label, dtype=torch.float32)
        return input_ids, attention_mask, label


############################
# 学習ループ
############################

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    total_count = 0

    criterion = nn.BCEWithLogitsLoss()

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    avg_loss = total_loss / total_count
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0

    criterion = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def _loop():
        nonlocal total_loss, total_count, total_correct
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            total_correct += (preds == labels).sum().item()

    _loop()
    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc


def save_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


############################
# エントリポイント
############################

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

    # 5桁までの数字を扱う前提で max_len=6
    tokenizer = DigitTokenizer(max_len=6)
    model = AhoTransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=tokenizer.max_len,
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

                input_ids, attention_mask = tokenizer.batch_encode_numbers([num])
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    logit = model(input_ids, attention_mask)[0]
                    prob = torch.sigmoid(logit).item()
                    pred = prob >= 0.5
                    rule = is_aho_number(num)
                    correct = rule == pred

                status = " / 正解" if correct else ""
                print(
                    f"モデル判定: {num} -> {'Aho' if pred else 'Not Aho'} "
                    f"(p={prob:.3f}) / ルール: {rule}{status}"
                )
        else:
            n = 1000
            test_numbers = [random.randint(10000, 99999) for _ in range(n)]
            input_ids, attention_mask = tokenizer.batch_encode_numbers(test_numbers)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).cpu().numpy()

            print("\n=== 判定結果 ===")
            correct = 0
            for n, p, y in zip(test_numbers, probs.cpu().numpy(), preds):
                tag = "Aho" if y == 1.0 else ""
                rule = is_aho_number(n)
                status = " / 正解" if rule == y else ""
                if rule == y:
                    correct += 1
                print(f"{n:3d} -> {tag} (p={p:.3f}, ルール={rule}{status})")
            total = len(test_numbers)
            acc = correct / total if total > 0 else 0.0
            print(f"\n正解率: {acc*100:.2f}% ({correct}/{total})")
                    
    else:
        # 学習ループ実行

        # ここで扱う数字の範囲を決める（講義時間に合わせて調整）
        # 1〜40000 を学習、40001〜50000 を検証に使う例
        train_numbers = list(range(1, 40001))
        val_numbers = list(range(40001, 50001))

        train_dataset = AhoNumberDataset(train_numbers, tokenizer)
        val_dataset = AhoNumberDataset(val_numbers, tokenizer)

        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=256, shuffle=False, num_workers=0
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        writer = SummaryWriter()
        num_epochs = 100
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            print(
                f"Epoch {epoch:02d} | "
                f"train_loss = {train_loss:.4f} | "
                f"val_loss = {val_loss:.4f} | "
                f"val_acc = {val_acc*100:.2f}%"
            )
            if epoch % 10 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pth")
                save_checkpoint(ckpt_path, model, optimizer, epoch)
                print(f"Saved checkpoint: {ckpt_path}")
