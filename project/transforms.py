from torchvision import transforms as tf


class Transforms():

    @staticmethod
    def __composer() -> list:
        """変換リスト生成

        https://pytorch.org/vision/stable/transforms.html#generic-transforms

        Returns:
            list: 変換リスト
        """
        x: list = []
        x_appned = x.append
        # x_appned(tf.Resize((224, 224)))
        x_appned(tf.ToTensor())
        x_appned(tf.Normalize((0.5, ), (0.5, )))
        # x_appned(tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        # x_appned(tf.CenterCrop(size))
        # x_appned(tf.ColorJitter(sbrightness=0, contrast=0, saturation=0, hue=0ize))
        # x_appned(tf.FiveCrop(size))
        # x_appned(tf.Grayscale(num_output_channels=1))
        # x_appned(tf.Pad(padding, fill=0, padding_mode='constant'))
        return x

    @staticmethod
    def compose() -> tf.transforms.Compose:
        """画像変換

        Returns:
            tf.transforms.Compose: 画像変換チェーン
        """
        return tf.Compose(Transforms.__composer())
