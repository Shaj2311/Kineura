# Kineura
<img width="1264" height="711" alt="kineuraLogo" src="https://github.com/user-attachments/assets/40d75077-cdd4-47bb-b32a-8aa163fece64" />

## Description
_Kineura_ is a deep learning model designed to predict and regenerate missing frames from a video using learned linear extrapolation. This is useful in cases like bandwidth-efficient video streaming and quality enhancement of footage captured by IoT devices or CCTV cameras with hardware limitations.

## Architecture
The project uses a standard U-Net architecture to downsample two consecutive input frames and upsampling them to generate a third output frame. The architecture also utilizes skip connections to maintain the appearance of static entities such as the background.
<img width="1408" height="768" alt="Gemini_Generated_Image_yxk03syxk03syxk0(1)" src="https://github.com/user-attachments/assets/f59fb4f8-8694-4802-a92d-cad3823460ee" />

## Dataset
The dataset used to train this model is the [Vimeo90K Triplet Dataset](https://www.kaggle.com/datasets/chenshu123/vimeo-triplet) downloaded from Kaggle.

## Performance
Upon testing the model, we generated the following performance metrics:
| MSE | PSNR | SSIM | LPIPS |
| --- | ---- | ---- | ----- |
| 0.0020 | 28.3759 | 0.896 | 0.1723 |

## Example Frames
### Example 1
<figure>
<img width="448" height="256" alt="im3" src="https://github.com/user-attachments/assets/d9c209ba-d9e8-4f6d-8e45-12ec8977f16d" />
<figcaption><i>Actual Frame</i></figcaption>
</figure>

<figure>
<img width="448" height="256" alt="output_frameSSIM" src="https://github.com/user-attachments/assets/cbbdcbe4-ec6b-426c-9fd0-1de0ce699b3e" />
<figcaption><i>Predicted Frame</i></figcaption>
</figure>

### Example 2
<figure>
<img width="448" height="256" alt="im3" src="https://github.com/user-attachments/assets/6734d8fd-a84d-4920-b93c-cbf6e31d9020" />
<figcaption><i>Actual Frame</i></figcaption>
</figure>

<figure>
<img width="448" height="256" alt="output_frameSSIM" src="https://github.com/user-attachments/assets/7cc3d1cc-4eb0-44e0-ad40-138e624ac864" />
<figcaption><i>Predicted Frame</i></figcaption>
</figure>


## Try It Out
```
git clone https://github.com/Shaj2311/Kineura
cd Kineura
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
