
PROMPT_GENERATE_CONTRAST_VLM = """Given a source image, the objective is to generate contrastive strategy and corresponding information for contrastive images. Each contrastive image should preserve most of the original visual features while diverging in one key detail, thereby forming a contrastive pair with the source image. The contrastive aspects should be selected from the following aspects.
1. Entities: Identifiable individuals, animals, objects, etc.
Specific aspects include:
(1) Entity Type: The specific, fine-grained category of the entity, such as Donald Trump, Labrador, Husky, Boeing 747, etc.
(2) Entity Attribute: Detailed visual or intrinsic characteristics of the entity, such as occupation, color, pattern, material, shape, etc.
(3) Entity Relationship: Spatial, interactive, or relational dynamics among entities, such as on top of, riding, shaking hands, husband and wife, friends, etc.
(4) Entity Emotion: Emotions of the entity, such as positive, negative, sad, and happy.

2. Scenes: The environment, background, and location depicted in the image.
Specific aspects include:
(1) Scene Type: The fine-grained categorization of the scene, such as market, lakeside, seaside, church interior, etc.
(2) Scene Attribute: Visual features or the state of the scene, such as crowded market, tranquil lakeside, clear blue sky, etc.

3. Actions and Events: Narrative events or behaviors performed by specific entities.
Specific aspects include:
(1) Category: Specific types of actions or events, such as chasing, dancing, and caressing.
(2) Event Attribute: Participants and characters that constitute the action or event, such as subject and object.

4. Style and Presentation: The visual rendering style and artistic representation of the image, such as photograph, painting, illustration, comic, ink painting, impressionism, or realism.

Instructions:
1. Identify Key Details: Based on the image, identify 3-4 key details in the image and output their corresponding aspects. Ignore details that are difficult to obtain contrastive images and less important details in the given detailed descriptions. These key details should preferably span different dimensions and aspects. For example:
Event Element: two Labradors are chasing the cat; Scene Attribute: The color of the lawn is green, hinting at summer 
2. Design Contrastive Strategies and Output Key Information: For each key detail, execute the following steps:
- Output current contrastive detail.
- Ouput aspect of current key detail from the above aspects.
- Design Contrastive Strategy: Based on the contrastive detail, propose a strategy for generating the contrastive image. The strategy must perturb the source image by changing current contrastive detail, such as shifting Entity Type from Labrador to Husky, Event Element from dogs chasing cat to cat chasing dogs, Entity Attribute from summer green lawn to land covered with snow in winter, or Style and Presentation from normal to retro classic style. Note that the contrastive strategy should be realistic and achievable.
- Output Image Editing Instruction: For each strategy, generate a clear, detailed editing instruction. An image editing tool will transform the original image into a contrastive image according to the editing instruction.

Example for the Output format: 
Key Details: Event Element: two Labradors are chasing the cat; Scene Attribute: The color of the lawn is green, hinting at summer

Contrastive Strategies:
Strategy-1:
Contrastive Detail: two Labradors are chasing the cat
Contrastive Aspect: Event Element
Contrastive Strategy: change the relationship of event element, from "two Labradors are chasing the cat" to "cat are chasing the two Labradors"
Image Editing Instruction: Replace the chaser role, making the cat the pursuer and the Labradors the targets.

Strategy-2:
Contrastive Detail: The color of the lawn is green, hinting at summer
Contrastive Aspect: Scene Attribute
Contrastive Strategy: change the scene from summer green lawn to winter snowfield
Image Editing Instruction: Transform the background from a green summer lawn into a snow-covered winter field.


Guidelines for Output:
Thoughts:
{{ Insert your reasoning and thinking here. }}
Output:
{{ Insert the output here. Note that follow the output format of example above. }}"""



PROMPT_FINE_CAPTION = """In the given {image_num} images, there are both similarities and subtle differences. The text below provides a coarse-grained caption for the first image, which does not capture the fine-grained content, especially the differences from the other images. Your task is to generate a fine-grained caption for each image, following the style of caption below.
Coarse-grained Caption of the First Image:
{caption}

Instructions:
1. Identify the most significant differences between the image pairs. Limit a maximum of two key differences for each image pair. If one detail difference between two images is significant, while the other is minor, then the less noticeable difference can be ignored. Avoid noting minor discrepancies. Focus on the Entity (Type, Attribute, Relationship), Scene (Type, Attribute, Spatial Relationship), Event/Action (Category, Element, Attribute), Emotion and Mood, Style and Presentation. If the images depict sports such as surfing, skiing, or skateboarding, describle the specific posture.
2. Summarize all the key differences into key aspects of differences.
3. Based on the key aspects summarized in above step, identify the specific key features of each individual image. Avoid referencing elements not visible in the image, or using vague phrases like "different from the first image." Instead, describe only what is explicitly shown.
4. For each image, following the style of the given coarse-grained caption, provide a fine-grained and fluent caption. The caption to ensure compliance with criteria below.
5. Translate the English captions into Chinese.
Separating each caption with "\n" and adding a index (such as 1. or 2.) before each caption.
For example:
Key Differences:
The difference between the first and second images is the chasing relationship between the Labradors and the cat, where the Labradors are the chasers in the first image, and the cat is the chaser in the second image. The difference between the first and third images is the environment, where the first image features a summer green lawn, while the third image showcases a winter snow-covered lawn.
Key Aspects of All Images:
The chasing relationship between the Labradors and the cat; the environment (summer green lawn, winter snow lawn).
Key Details of Each Image:
Image 1: Labradors are the chasers, and the environment is a summer green lawn. Image 2: The cat is the chaser, and the environment is a summer green lawn. Image 3: Labradors are the chasers, and the environment is a winter snow-covered lawn.
Image Captions: 
1.Two Labradors are happily chasing a cat on the summer green lawn.
2.A cat are happily chasing two Labradors on the summer green lawn.
3.Two Labradors are happily chasing a cat on the winter snow lawn.
Chinese Image Captions: 
1.两只拉布拉多犬在夏日绿茵茵的草坪上快乐地追逐一只猫。
2.一只猫在夏日绿茵茵的草坪上快乐地追逐两只拉布拉多犬。
3.两只拉布拉多犬在冬日积雪覆盖的草坪上快乐地追逐一只猫。

The caption for image must meet the following criteria:
(1) Consistency with Content: The caption for each image should reflect its content. While not every detail needs to be covered, it must accurately represent the essential elements of the image.
(2) Highlighting Differences: The caption must emphasize the differences between this image and the others. At least one distinguishing feature should be included, ensuring the caption reflects a divergence from the other images.
(3) Image-Specific Details: The caption must focus on the details present within the image itself. Avoid referencing elements not visible in the image, or using vague phrases like "different from the first image" or "the background are different." Instead, describe only what is explicitly shown. For example, use "There is only a white computer on the table" rather than "There is a white computer on the table, but no books.", Use "The dog is looking to the side" instead of "The dog is not looking at the camera,", use "An empty road background" rather than "There are no black cars in the road background."
(4) Based on the first three criteria, the captions of different images should reflect the differences between the images.

Guidelines for Output:
- Be succinct.
- Structure your response as follows:
Key Differences:
{{ Insert the key differences between images here. }}
Key Aspects of All Images:
{{ Insert the key aspects of all images here. }}
Key Details of Each Image:
{{ Insert key details of each image here. }}
English Image Captions: 
{{ Insert the fine-grained English image captions of all images here. The number of captions must equal to {image_num}. Note that follow the output format of example above. Avoid referencing elements not visible in the image, or using vague phrases like "different from the first image" or "the background are different." }}
Chinese Image Captions: 
{{ Insert the Chinese image captions here. }}"""

