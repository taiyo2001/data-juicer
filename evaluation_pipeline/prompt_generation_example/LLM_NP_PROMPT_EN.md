## LLM Instruction for Negative Prompt Generation

### 1. Objective

The LLM must generate a **Custom Negative Prompt** designed to precisely suppress failures in the Image Generation Model related to **"faithful reproduction of details," "accurate counting and placement of objects,"** and **"complex spatial relationships."** The output must be a single, concatenated string of weighted keywords.

### 2. Generation Rules and Categories

The Negative Prompt must be constructed based on the following categories:

  * **Category A (Fundamental)**: Ensures image quality.
  * **Category B (Counting/Existence)**: Enforces the correct number and presence of objects and attributes.
  * **Category C (Attributes/Composition)**: Negates inaccurate poses, attributes, and background inconsistencies.

### 3. Few-Shot Learning Example

The LLM must learn the following pairs of **Input Prompt** and corresponding **Optimized Custom Negative Prompt (Ground Truth)**.

#### Input Prompt Example (Puppy Prompt)

```prompt
An overhead view of four labradoodle puppies, three puppies are sitting and one puppy is standing with its right paw resting against the white barrier at the bottom of the image. The puppies are on a light blue rug placed on a black floor. The puppy standing is a beige and white puppy with curly fur, dark eyes, a small nose, and a fluffy appearance, its paw extended. There is a black and white puppy sitting on its hind legs to the right, and to the left part of the image is another beige puppy sitting on its hind legs as well. Directly behind the standing puppy, in the upper part of the image, is another light cream colored puppy sitting on its hind legs, looking toward the bottom right corner of the image. The three puppies in the front are looking up, the puppy behind them is looking toward the bottom right corner of the image. There is a blue plush toy in the bottom right corner of the image underneath the black puppy. The rug the puppies are on is not laying completely flat on the ground, its unintentionally folded up in some areas and folded over itself in the top right corner of the image. The background consists of a light blue rug placed on a black floor, with the rug showing some unintentional folds and overlaps. A blue plush toy is visible in the bottom right corner under the black puppy. The image is well-lit with soft, even lighting, suggesting an indoor setting with artificial light sources. The light appears to be front-lit, as there are no harsh shadows on the puppies. The style of the image is a realistic photo. The beige and white puppy standing with its right paw resting against the white barrier is in front of the light cream colored puppy sitting on its hind legs in the back. The black and white puppy sitting on its hind legs to the right is to the right of the beige and white puppy standing with its right paw resting against the white barrier. The beige puppy sitting on its hind legs to the left is to the left of the beige and white puppy standing with its right paw resting against the white barrier. The light cream colored puppy sitting on its hind legs in the back is behind the beige and white puppy standing with its right paw resting against the white barrier. The black and white puppy sitting on its hind legs to the right is next to the beige puppy sitting on its hind legs to the left.
```

#### Custom Negative Prompt to be Generated (Ground Truth)

```
(low quality, worst quality:1.4), blurry, noisy, jpeg artifacts, deformed, disfigured, bad anatomy, ugly, (1 puppy, 2 puppies, 3 puppies, 5 puppies:1.5), (wrong number of puppies:1.3), (missing puppies:1.4), (missing toy:1.3), (missing barrier:1.3), (all sitting:1.5), (all standing:1.5), (wrong pose:1.2), (all one color:1.3), (flat rug, smooth rug, neat rug:1.4), (simple background:1.2), (cropped:1.1), (out of frame:1.1)
```

### 4. Task and Output Format

Strictly apply the structure and rules from the Few-Shot example above, and generate a Custom Negative Prompt in English for the following Positive Prompt.

**Output Format:** Output the concatenated Negative Prompt for all categories as a **single string** of approximately 20 to 200 characters.

**Reason step-by-step, and input the final answer within \boxed{}.**

**Positive Prompt:**
