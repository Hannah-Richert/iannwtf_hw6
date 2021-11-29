from math import ceil, floor
import tkinter
import tensorflow_datasets as tfds
import tensorflow as tf


class ReceptiveField:
    """
    Compute the rfield for a given sequence of tensorflow layers
    Functions:
        - __init__: constructor
        - add_rfield: add a rfield
        - abstract_fields: translate the position of all fields into a higher layer
        - mutate_center: translate the size of all fields into a higher layer
        - compute_rfield_from_raw_values: compute the rfield given raw values
        - compute_rfield_from_layers: compute rfield given a list of tensorflow layers
        - plot: plot the rfield onto a base image
        - _from_rgb: convert rgb color into tkinter color (source: https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter/51592104)

    """

    def __init__(self):
        """
        Initialize the list of rfields
        """
        self.r_field = []

    def add_rfield(self, r_field):
        """
        Add a rfield to the list of current rfields
            Args:
                - r_field: <list<int>> List of coordinates.
        """
        self.r_field.append(r_field)

    def abstract_fields(self, kernel, padding, stride, size):
        """
        Translate position of all currently stored receptive fields given the kernel_size, padding and stride

            Args:
                - kernel: <list<int>> Kernel size of the layer in which to translate
                - padding: <string> "same" or "valid"
                - strides: <list<int>> Stride size of the layer in which to translate
        """
        new_r_fields = []
        for field in self.r_field:
            if padding == "valid":

                # rfield with the translated position
                new_rfield = []

                # go through each dimension of the kernel, field and stride
                # and calculate the new position for each dimension
                for dim_kernel, dim_field, dim_stride in zip(kernel, field, stride):
                    if dim_kernel % 2 == 1:
                        new_rfield.append(
                            dim_stride*dim_field+floor(dim_kernel/2))
                    else:
                        new_rfield.append(
                            dim_stride*dim_field+floor(dim_kernel/2-1))

            elif padding == "same":
                new_rfield = []
                # go through each dimension of the kernel, field and stride
                # and calculate the new position for each dimension
                for dim_field, dim_stride in zip(field, stride):
                    new_rfield.append(dim_stride*dim_field)

            # translate the scale of each new rfield
            for new_field in self.mutate_center(new_rfield, kernel, size):
                if not new_field in new_r_fields:
                    new_r_fields.append(new_field)

        self.r_field = new_r_fields

    def mutate_center(self, center, kernel, size):
        """
        Each list of fields over which the kernel iterates produces a new field in the next layer.
        In order to get the corresponding receptive fields of a higher layer 
        (which were used by the kernel for producing a new field), 
        we have to reverse this process given the center on which the kernel produced the new field.
            Args: 
                - center: <list<int>> The rfield which represents the center
                - kernel: <list<int>> Kernel size of the layer in which to mutate
                - size: <list<int>> Size of the layer in which to translate
            Returns:
                - to_mutate: <list<list<int>>> List of mutated/translated values
        """
        to_mutate = [center]
        # iterate over all dimensions of the kernel
        for dim, dim_size in enumerate(kernel):

            # all mutations given this dimension
            extend_to_mutate = []
            for field in to_mutate:
                for i in range(1, floor(dim_size/2)+1):

                    # extend field to the left
                    if dim_size % 2 == 1:
                        subtract = field.copy()
                        subtract[dim] = subtract[dim]-i
                    # if the kernel in this dimension has an even size, the kernel has no center
                    # but rather 2 centers in this dimension. Because we take the leftmost of
                    # these two centers as its center, we have to take one less step to the "left".
                    # so we add 1 to the step to the "left".
                    else:
                        subtract = field.copy()
                        subtract[dim] = subtract[dim]-i+1

                    # extend field to the right
                    add = field.copy()
                    add[dim] = add[dim]+i

                    # check if field is not out of bounds
                    if not(subtract[dim] < 0 or subtract[dim] > size[dim]-1):
                        # add field if it isn't already contained in any list
                        if not (subtract in to_mutate and subtract in extend_to_mutate):
                            extend_to_mutate.append(subtract)

                    # check if field is not out of bounds
                    if not(add[dim] < 0 or add[dim] > size[dim]-1):
                        # add field if it isn't already contained in any list
                        if not (add in to_mutate and add in extend_to_mutate):
                            extend_to_mutate.append(add)

            to_mutate.extend(extend_to_mutate)
        return to_mutate

    def compute_rfield_from_raw_values(self, kernels, paddings, strides, img_size):
        """
        Compute the rfield from the raw values
            Args:
                - kernels: <list<list<int>>> List of kernels
                - padding: <list<list<int>>> List of paddings
                - strides: <list<string>> List of strides
                - img_size: <list<int>> List of integers. Size of the image for each dimension
        """
        sizes = [img_size]
        for kernel, padding, stride in zip(kernels, paddings, strides):

            # compute the sizes for each layer
            if padding == "valid":
                new_size = []
                for dim_kernel, dim_stride, dim_size in zip(kernel, stride, sizes[-1]):
                    new_size.append(ceil((dim_size-dim_kernel+1)/dim_stride))
                sizes.append(new_size)
            elif padding == "same":
                new_size = []
                for dim_stride, dim_size in zip(stride, sizes[-1]):
                    new_size.append(floor(dim_size/dim_stride))
                sizes.append(new_size)

        kernels.reverse()
        paddings.reverse()
        strides.reverse()
        # delete the last size because its the output size and we don't need that
        sizes.__delitem__(-1)
        sizes.reverse()
        for kernel, padding, stride, size in zip(kernels, paddings, strides, sizes):
            self.abstract_fields(kernel, padding, stride, size)

    def compute_rfield_from_layers(self, layers, img_size):
        """
        Compute rfield given list of tensorflow layers
            Args:
                - layers: <list> List of tensorflow layers
                - img_size: <list<int>>. Size of the image for each dimension
        """
        kernels = []
        paddings = []
        strides = []
        for layer in layers:
            if(isinstance(layer, tf.keras.layers.Conv2D)):
                kernels.append(list(layer.kernel_size))
                paddings.append(layer.padding)
                strides.append(list(layer.strides))
            elif(isinstance(layer, tf.keras.layers.AveragePooling2D) or isinstance(layer, tf.keras.layers.MaxPool2D)):
                kernels.append(list(layer.pool_size))
                paddings.append(layer.padding)
                if layer.strides is None:
                    strides.append(list(layer.pool_size))
                else:
                    strides.append(list(layer.strides))
                    
        self.compute_rfield_from_raw_values(
            kernels, paddings, strides, img_size)

    def plot(self, image, size=[1, 1], offset=[0, 0]):
        """
        Plot the rfield given a base image
            Args:
                - image: <list<list<list<int>>>> Has to have the shape: (_,_,1), where 1 represents a list of a single grayscale value
                - size: <[x,y]>. Size of each pixel for each dimension
                - offset: <[x,y]>. Offset of the overall image
        """
        root = tkinter.Tk()
        root.geometry("600x600")
        canvas = tkinter.Canvas(root)
        canvas.pack()

        for count_x, x in enumerate(image):
            for count_y, y in enumerate(x):
                g = y[0]
                canvas.create_rectangle(count_x*size[0]+offset[0], count_y*size[1]+offset[1], count_x*size[0]+size[0]+offset[0], count_y *
                                        size[1]+size[1]+offset[1], fill=self._from_rgb((g, g, g)), outline=self._from_rgb((g, g, g)))

        for coordinate in self.r_field:
            count_x = coordinate[0]
            count_y = coordinate[1]
            canvas.create_rectangle(count_x*size[0]+offset[0], count_y*size[1]+offset[1], count_x*size[0]+size[0]+offset[0], count_y *
                                    size[1]+size[1]+offset[1], outline=self._from_rgb((0, 255, 0)))

        root.mainloop()

    def _from_rgb(self, rgb):
        """
        translates an rgb tuple of integers to hex
            Args:
                - rgb: <tuple> tuple of rgb values

            Returns:
                - rgb: <string> rgb value translated to hex
        """
        return "#%02x%02x%02x" % rgb
