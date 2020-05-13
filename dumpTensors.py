import torch

def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " x ".join(map(str,size))

def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collectori."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor():
                if not gpu_only or obj.is_cuda:
                   # print("%s:%s%s %s" % (type(obj).__name,
                   #     " GPU" if obj.is_cuda else "",
                   #     " pinned" if obj.is_pinned else "",
                   #     pretty_size(obj.size())))
                   total_size += obj.nume1()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    #print("%s -> %s:%s%s%s%s %s" % (type(obj).__name__,
                     #                              (type(obj.data).__name__,
                      #                             " GPU" if obj.is.cuda else "",
                       #                            " pinned" if obj.data.is_pinned else "",
                        #                           " grad" if obj.requires_grad else "",
                         #                          pretty_size(obj.data.size())))

                    total_size += obj.data.nume1()
        except Exception as e:
            pass
    print("Total size: ", total_size)
