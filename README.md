# ShinkaiGAN

ShinkaiGAN is a deep learning model designed to transform sketch images into beautiful anime scenes inspired by the style of Makoto Shinkai. This model utilizes a Hybrid Perception Block U-Net architecture to achieve high-quality image-to-image translation. In order to stabilize training process, we adopt the progressive training techniques as Karras, et. al. proposed to train ProGAN and StyleGANs.

## Model Architecture

The core of ShinkaiGAN is based on UNet with the Hybrid Perception Block architecture. 
![image](https://github.com/echelon2718/ShinkaiGAN/assets/92637327/12049550-5936-48ed-a0be-a2cb616c11ef)

## Dataset

The model is trained on a custom dataset that includes:
- High-resolution anime scenes from various Makoto Shinkai movies.
- Corresponding sketch images manually created or extracted using edge detection algorithms.

## Usage

To use ShinkaiGAN, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/ShinkaiGAN.git
    cd ShinkaiGAN
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run Training:**

    ```bash
    python train.py \
      --src_dir "/path/to/source/directory" \
      --tgt_dir "/path/to/target/directory" \
      --lvl1_epoch 10 \
      --lvl2_epoch 20 \
      --lvl3_epoch 30 \
      --lvl4_epoch 40 \
      --lambda_adv 1.0 \
      --lambda_ct 0.1 \
      --lambda_up 0.01 \
      --lambda_style 0.01 \
      --lambda_color 0.001 \
      --lambda_grayscale 0.01 \
      --lambda_tv 0.001 \
      --lambda_fml 0.01
    ```


## Results

Here are some examples of sketch-to-anime transformations using ShinkaiGAN:

| Sketch | Anime Scene |
|--------|--------------|
| ![image](https://github.com/echelon2718/ShinkaiGAN/assets/92637327/34cabd2f-0a11-4c86-9326-7d8695aebf04) | ![image](https://github.com/echelon2718/ShinkaiGAN/assets/92637327/d3f0fcb1-f661-4f5e-9e12-98ea00d80edd) |
| ![image](https://github.com/echelon2718/ShinkaiGAN/assets/92637327/29997a10-4179-46b8-8db5-557d65194c94) | ![image](https://github.com/echelon2718/ShinkaiGAN/assets/92637327/162506df-f66c-40d8-935b-a077169cbef5) |

## Contributing

We welcome contributions to improve ShinkaiGAN. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
