import torchvision.datasets as datasets
from dataset_utils import *

class IsNewFile:
    def __init__(self):
        self.new_images = []
        self.ds_idx = []
        self.nn_name = None
        self.ds_util = None
        self.false_cnt = 0
        self.true_cnt = 0

    def __call__(self, path):
        if self.nn_name is None:
          print("path:", path)
          dataset_info = path.split("/")
          last_iteration = len(dataset_info)
          for i, ds_dir in enumerate(dataset_info):
            if ds_dir == "FUNC":
              last_iteration = i + 1
            if last_iteration == i:
              self.nn_name = ds_dir
              print("nn_name:", self.nn_name)
              self.ds_util = DatasetUtils(self.nn_name, "FUNC")
              break
          self.new_images,self.ds_idx = self.ds_util.get_dataset_images(mode="FUNC", nn_name=self.nn_name, position="NEXT")
          print("num new_images:", len(self.new_images))
          # print("new_images:", self.new_images)
          self.new_images = tuple(self.new_images)
        if path in self.new_images:
#          print("ds_idx:", self.ds_idx)
#          for directory, num_images in self.idx_file:
#            print("directory, path:", directory, path)
#            if directory == path:
#              if num_images == 0:
#                print("filtering empty directory from valid files:", path)
#                return False
#              print("non-empty directory is valid:", path)
#              break
          self.true_cnt += 1
          return True
        else:
          # print(path)
          self.false_cnt += 1
          if self.false_cnt % 10 == 0:
              print("count False, True", self.false_cnt , self.true_cnt, path)
          # print("False path:", path)
          return False


# based on:
# https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#
# The goal of this class is to remap data collected for a particular function to a more generalized
# set of metadata. For example, a particular function may not use all robot actions (e.g., the arm
# or reverse) but the NN is consistently trained across all possible robot actions.
# Another example is "automatic mode", where the primitive actions can be mapped to to NOOPs
# if determined programatically.
# 
# only_new_images: root directory 
class ImageFolder2(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            app_name,
            app_type,
            transform = None,
            target_transform = None,
            # loader: Callable[[str], Any] = torchvision.default_loader,
            is_valid_file = None,
            full_action_set = None,
            remap_to_noop = None,
            only_new_images = None,
    ):
        # super(datasets.ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
        self.app_type = app_type
        if self.app_type == "FUNC":
            self.app_name = None
            self.nn_name = app_name
            self.ds_util = DatasetUtils(self.nn_name, self.app_type)
        else:
            self.app_name = app_name
            self.nn_name = None
            self.ds_util = DatasetUtils(self.app_name, self.app_type)
        # From original code:
        # self.classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        # self.classes.sort()
        # self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # self.imgs= [image_path, class_index]
        if full_action_set is not None:
          # full_action_set.sort()  # full_action_set is a sorted tuple
          self.classes = full_action_set
          self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
          print("new:", self.class_to_idx)

        if full_action_set is not None or remap_to_noop is not None:
          old_classes = self.classes
          old_class_to_idx = self.class_to_idx
          print("old:", old_class_to_idx)

        # print("calling super(ImageFolder2)")
        # if True:
        #   print("ARD: find exception calling super(ImageFolder2)")
        #   File "/usr/local/lib/python3.6/dist-packages/torchvision-0.4.0a0+d31eafa-py3.6-linux-aarch64.egg/torchvision/datasets/folder.py", line 97, in __init__
        # TypeError: can only join an iterable
        # 
        # Misleading error:
        # If you make an Imagefolder dataset with no samples in the directory and a is_valid_file. 
        # Because the extensions variable is set to none 
        # because is_valid_file which causes the TypeError.
        #if True:
        try:
          if only_new_images is not None and only_new_images:
            print("only new images", root)
            super(ImageFolder2, self).__init__(root, 
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=IsNewFile())
          else:
            print("all images: ", root, is_valid_file)
            super(ImageFolder2, self).__init__(root, transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        # else:
        except Exception as e:
          self.imgs = []
          # print("exception calling super(ImageFolder2):", e.message)
          print("exception calling super(ImageFolder2):", e)
          return 
        self.imgs = self.samples
        print("Number of images:", len(self.imgs))

        if remap_to_noop is not None:
          # NOOP must be in class list
          noop_idx = self.classes.index("NOOP")

          old_class_idx = []
          for old_class in remap_to_noop: 
              old_class_idx.append( old_classes.index(old_class))

        if only_new_images is not None and only_new_images:
          # new_images = ds_util.new_images(root)
          new_images,ds_idx = self.ds_util.get_dataset_images(mode="FUNC", nn_name=self.nn_name, position="NEXT", exclude_FPEN=True)

          img_lst = []
          for i, [image_path, old_class_index] in enumerate(self.imgs): 
            # for j, [image_path2, class_index2] in enumerate(new_images): 
            #   if image_path == image_path2:
                if image_path in new_images:
                  img_lst.append(self.imgs[i])
          print("original number of images in dataset:", len(self.imgs))
          print("number of new images in dataset     :", len(img_lst))
          print("number of new images in dataset idx :", len(new_images))
          print("ds_idx:", ds_idx)
          # if len(img_lst) == 0:
          #   print("self.imgs:", self.imgs)
          #   print("new_imgs: ", new_images)
          self.imgs = img_lst

        # currently we're using full_action_set everywhere.
        # In the past, we allowed individual NNs to train on only a subset of actions.
        # We will probably need to do so again in the future. 
        # ARD: Changing 7/5/21
        if full_action_set is not None or remap_to_noop is not None or only_new_images is not None:
          for i, [image_path, old_class_index] in enumerate(self.imgs): 
              if (remap_to_noop is not None and old_class_index in old_class_idx):
                  item = image_path, noop_idx
                  self.imgs[i] = item
                  print(self.imgs[i], old_class_index)
              elif (full_action_set is not None and
                    old_classes[old_class_index] != self.classes[old_class_index]):
                  # self.classes should be a superset of the old_classes,
                  # but the sorting can reorder the idx.
                  print("old_class_index", old_class_index)
                  print("old_classes", old_classes)
                  print("class_to_idx", self.class_to_idx)
                  print("self.classes", old_classes[old_class_index], self.classes[old_class_index])
                  new_act = old_classes[old_class_index]
                  print("full_action_set,new_act:", full_action_set, new_act)
                  if new_act in full_action_set:
                    new_item  = image_path, self.class_to_idx[old_classes[old_class_index]]
                    self.imgs[i] = new_item
                    print(self.imgs[i], old_class_index)
                  else:
                    print("discarded from training:", self.imgs[i])

    def all_images_processed(self, mode, nn_name):
          return self.ds_util.all_indices_processed(mode, nn_name)

    def save_images_processed(self, mode, nn_name):
          # new_images = ds_util.new_images(root)
          self.ds_util.dataset_images_processed(mode, nn_name)
