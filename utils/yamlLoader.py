# 基于类的解决方案，使用元类注册自定义构造函数。
import os

import yaml
from loguru import logger

class IncludeLoader(yaml.Loader):                                                 
    """                                                                           
    yaml.Loader subclass handles "!include path/to/foo.yml" directives in config  
    files.  When constructed with a file object, the root path for includes       
    defaults to the directory containing the file, otherwise to the current       
    working directory. In either case, the root path can be overridden by the     
    `root` keyword argument.                                                      

    When an included file F contain its own !include directive, the path is       
    relative to F's location.                                                     

    Example:                                                                      
        YAML file /home/frodo/one-ring.yml:                                       
            ---                                                                   
            Name: The One Ring                                                    
            Specials:                                                             
                - resize-to-wearer                                                
            Effects: 
                - !include path/to/invisibility.yml                            

        YAML file /home/frodo/path/to/invisibility.yml:                           
            ---                                                                   
            Name: invisibility                                                    
            Message: Suddenly you disappear!                                      

        Loading:                                                                  
            data = IncludeLoader(open('/home/frodo/one-ring.yml', 'r')).get_data()

        Result:                                                                   
            {'Effects': [{'Message': 'Suddenly you disappear!', 'Name':            
                'invisibility'}], 'Name': 'The One Ring', 'Specials':              
                ['resize-to-wearer']}                                             
    """                                                                           
    def __init__(self, *args, **kwargs):                                          
        super(IncludeLoader, self).__init__(*args, **kwargs)                      
        self.add_constructor('!include', self._include)                           
        if 'root' in kwargs:                                                      
            self.root = kwargs['root']                                                                  
        else:                                                                     
            self.root = os.path.curdir                                            

    def _include(self, loader, node):                                    
        oldRoot = self.root                                              
        filename = os.path.join(self.root, loader.construct_scalar(node))
        self.root = os.path.dirname(filename)                         
        data = yaml.load(open(filename, 'r'),Loader=yaml.Loader)                            
        self.root = oldRoot                                              
        return data                                                      


def get_yaml_data(fileDir):
    
    """
    读取 test.yaml 文件内容
    :param fileDir:
    :return:
    """
    # 1、在内存里加载这个文件
    # f = open(fileDir, 'r', encoding='utf-8')
    with open(fileDir,'r', encoding='utf-8') as f:
        # data = yaml.load(f, yaml.Loader)
        data = yaml.load(f, Loader=IncludeLoader)
    # print(data)
    return data

if __name__ == '__main__':
    # 项目路径
    yaml_path="../config/config.yaml"
    info = get_yaml_data(yaml_path)
    print(info)

# 输出：
# {'a': 1,'b': [2, 3],'c': [10, [100,200, 300]]}
