## ネガティブプロンプト生成のためのLLM指示書

### 1. 目的

LLMは、画像生成モデルが苦手とする「詳細の忠実な再現」「オブジェクトの正確な計数と配置」「複雑な空間的関係性」の失敗をピンポイントで抑制する**カスタムネガティブプロンプト**を生成する。出力は、ウェイト付きのキーワードを連結した単一の文字列であること。

### 2. 生成ルールとカテゴリー

以下のカテゴリーに基づき、ネガティブプロンプトを構成する。

* **カテゴリーA (基本)**: 画像品質の担保。
* **カテゴリーB (計数/存在)**: オブジェクトの数と存在、属性の強制。
* **カテゴリーC (属性/構図)**: 複雑なポーズ、属性、背景の不正確さを否定。

### 3. Few-Shot学習例

LLMは以下の**入力プロンプト**と、それに対応する**最適化されたカスタムネガティブプロンプト（正解）**のペアを学習せよ。

#### 入力プロンプト例（子犬のプロンプト）

```prompt
An overhead view of four labradoodle puppies, three puppies are sitting and one puppy is standing with its right paw resting against the white barrier at the bottom of the image. The puppies are on a light blue rug placed on a black floor. The puppy standing is a beige and white puppy with curly fur, dark eyes, a small nose, and a fluffy appearance, its paw extended. There is a black and white puppy sitting on its hind legs to the right, and to the left part of the image is another beige puppy sitting on its hind legs as well. Directly behind the standing puppy, in the upper part of the image, is another light cream colored puppy sitting on its hind legs, looking toward the bottom right corner of the image. The three puppies in the front are looking up, the puppy behind them is looking toward the bottom right corner of the image. There is a blue plush toy in the bottom right corner of the image underneath the black puppy. The rug the puppies are on is not laying completely flat on the ground, its unintentionally folded up in some areas and folded over itself in the top right corner of the image. The background consists of a light blue rug placed on a black floor, with the rug showing some unintentional folds and overlaps. A blue plush toy is visible in the bottom right corner under the black puppy. The image is well-lit with soft, even lighting, suggesting an indoor setting with artificial light sources. The light appears to be front-lit, as there are no harsh shadows on the puppies. The style of the image is a realistic photo. The beige and white puppy standing with its right paw resting against the white barrier is in front of the light cream colored puppy sitting on its hind legs in the back. The black and white puppy sitting on its hind legs to the right is to the right of the beige and white puppy standing with its right paw resting against the white barrier. The beige puppy sitting on its hind legs to the left is to the left of the beige and white puppy standing with its right paw resting against the white barrier. The light cream colored puppy sitting on its hind legs in the back is behind the beige and white puppy standing with its right paw resting against the white barrier. The black and white puppy sitting on its hind legs to the right is next to the beige puppy sitting on its hind legs to the left.
```

#### 生成すべきカスタムネガティブプロンプト（正解）

```
(low quality, worst quality:1.4), blurry, noisy, jpeg artifacts, deformed, disfigured, bad anatomy, ugly, (1 puppy, 2 puppies, 3 puppies, 5 puppies:1.5), (wrong number of puppies:1.3), (missing puppies:1.4), (missing toy:1.3), (missing barrier:1.3), (all sitting:1.5), (all standing:1.5), (wrong pose:1.2), (all one color:1.3), (flat rug, smooth rug, neat rug:1.4), (simple background:1.2), (cropped:1.1), (out of frame:1.1)
```

### 4. タスクと出力形式

上記Few-Shot例の構造とルールを厳密に適用し、以下のポジティブプロンプトに対してカスタムネガティブプロンプトを英語で生成せよ。

出力形式: 20〜200文字程度の**単一の文字列**として、すべてのカテゴリーを連結したネガティブプロンプトを出力する。

**段階的に推論し、最終的な答えを \boxed{} 内に入力してください。**

**ポジティブプロンプト:**
