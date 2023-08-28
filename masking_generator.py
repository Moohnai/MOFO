import numpy as np

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 
    

class TubeMaskingGenerator_BB:
    def __init__(self, input_size, mask_ratio, mask_ratio_BB):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.mask_ratio = mask_ratio
        self.mask_ratio_BB = mask_ratio_BB

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self, bb):
        #find the patches that are inside the bounding box
        mask_per_frame = np.zeros(self.num_patches_per_frame)
        i = 0
        index = []
        for j in range(self.height):
            for k in range(self.width):
                x1_tube = j*16
                x2_tube = j*16 + 16
                y1_tube = k*16
                y2_tube = k*16 + 16
                # if the bounding box is inside the tube patch
                if not ((bb[i][0] > x2_tube or bb[i][2] < x1_tube) and (bb[i][1] > y2_tube or bb[i][3] < y1_tube)):
                    mask_per_frame[j*self.width + k] = 1
                    index.append(j*self.width + k)
                
        # create a zero for whole patches
        f = np.zeros(self.num_patches_per_frame)
        # shuffle the index
        np.random.shuffle(index)
        # select 90% of the patches in index
        cap_index = min(self.num_masks_per_frame, int(len(index)*self.mask_ratio_BB))#0.9
        selected_index = index[:cap_index]
        # set the mask to 0 for the selected patches
        for i in selected_index:
            f[i] = 1

        # total number of patches
        remaining_masks = self.num_masks_per_frame - len(selected_index)
        remaining_index = np.setdiff1d(np.arange(self.num_masks_per_frame), selected_index)

        # select the remaining masks randomly
        np.random.shuffle(remaining_index)
        for i in remaining_index[:remaining_masks]:
            f[i] = 1

        # mask_per_frame = np.hstack([
        #     np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
        #     np.ones(self.num_masks_per_frame),
        # ])
        # np.random.shuffle(mask_per_frame)
        mask = np.tile(f, (self.frames,1)).flatten()
        return mask 