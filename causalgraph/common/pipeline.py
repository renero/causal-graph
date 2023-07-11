"""

Pipeline class to define and run several execution steps.
(C) J. Renero, 2023

"""
import inspect
import types
from typing import Any, Dict, List, Union

from tqdm.auto import tqdm

from causalgraph.common import tqdm_params


# TODO: Eliminate the need to pass the host object to the pipeline

class Host:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def host_method(self):
        # print(f"  Into the method {self.host_method.__name__}")
        return ("Host.host_method")


class SampleClass:
    def __init__(self, param1=None, param2=False):
        self.param1 = param1
        self.param2 = param2
        self.fitted = False
        print(f"  > Into the init of class {self.__class__}")
        print(f"    > I got params: {self.param1}, {self.param2}")

    def fit(self):
        print(f"  > Into the fit of class {self.__class__}")
        print(f"    > I have params: {self.param1}, {self.param2}")
        self.fitted = True
        return self

    def method(self):
        print(f"  > Into method \'{self.method.__name__}\'")
        return "<Have been in SampleClass.method>"


def my_method(param1, param2):
    print(f"  > Into the method \'{my_method.__name__}\'")
    print(f"    > I got params: {param1}, {param2}")
    return f"my_method({param1} {param2})"


def method_with_object(obj):
    print(f"  > Into the method \'{method_with_object.__name__}\'")
    print(f"    > I got object: {obj}")
    return f"{obj.method()}>"


def m1(message="default_message"):
    return (f"m1_return_value={message}")


def m2(what):
    return (f"m2_return_value={what}")


what = "(argument for m1 and m2)"


# TODO: Eliminate the need to pass the host object to the pipeline
# TODO: COnsider the case when parameters are not specified and do not exist, but
#       the method does not need them because they are optional.
class Pipeline:
    """
    Pipeline class allows to define several execution steps to run sequentially.
    Pipeline is initialized with a host object that contains the parameters to be
    used in the execution steps.
    At each step, the pipeline can call either a function or a class. If a class
    is called, the pipeline will call the default method of the class. Such a default
    method must be specified in the pipeline constructor.
    If a function is called, it must be present globally or inside the host object.
    The pipeline can also create an attribute inside the host object with the value
    returned by the function or the fit method of the class.

    Example:
    --------
    >>> class Host:
    >>>     def __init__(self, param1, param2):
    >>>         self.param1 = param1
    >>>         self.param2 = param2
     
    >>> def my_method(param1, param2):
    >>>     return f"{param1}, {param2}"
     
    >>> host = Host(param1='Hello', param2='world')
    >>> pipeline = Pipeline(host, verbose=True, prog_bar=False)
    >>> steps = [
    >>>     'my_method',
    >>>     ('result1', 'my_method'),
    >>>     ('result2', 'my_method', {'param2': 'there!'})
    >>> ]
    
    >>> pipeline.run(steps)
    >>> print(host.result1)
    >>> print(host.result2)
    Hello, world
    Hello, there
    """

    def __init__(
            self, 
            host: type, 
            prog_bar: bool = True, 
            verbose: bool = False):
        """
        Parameters
        ----------
        host: object
            Object containing the parameters to be used in the execution steps.
        """
        self.host = host
        self._verbose = verbose
        self._prog_bar = prog_bar
        self._objects = {'host': self.host}

        # When passing a class to the pipeline, the pipeline will call the 
        # default method specified by _default_object_method, and will pass
        # the parameters specified by _default_method_params.
        self._default_object_method = None
        self._default_method_params = None

    def _get_step_components(self, step_name):
        """
        Get the parameters of a method.

        Parameters
        ----------
        step_name: str
            Name of the method.

        Returns
        -------
        list
            List of parameters of the method.
        """

        # Check if step_name is a string or a tuple. In the later case the tuple can
        # also include a return_value name and/or a dictionary with values.
        if type(step_name) is tuple:
            if len(step_name) == 3:
                return_value, step_call, step_arguments = step_name
            elif len(step_name) == 2 and type(step_name[1]) is dict:
                step_call, step_arguments = step_name
                return_value = None
            elif len(step_name) == 2:
                return_value, step_call = step_name
                step_arguments = []
            else:
                raise ValueError(
                    f"Tuple {step_name} must have 2 or 3 elements")
        else:
            step_call = step_name
            return_value = None
            step_arguments = []

        # Check if step_name is a method within Host, a method or a function in globals
        method = self._get_callable_method(step_call)
        arguments = inspect.signature(method).parameters
        step_parameters = {arg: arguments[arg].default for arg in arguments.keys()}

        # If step_parameters has 'self' as first key, remove it.
        if 'self' in step_parameters.keys():
            step_parameters.pop('self')

        return return_value, step_call, step_parameters, step_arguments

    def _get_callable_method(self, step_call):
        """
        Get the callable method from the host object or globals, the potential
        return value name to be stored and the arguments to be used in the method.

        Parameters
        ----------
        step_call: str
            Name of the method.

        Returns
        ------- 
        tuple
            Tuple containing the return value name, the callable method and the
            arguments to be used in the method.

        TODO: Allow recursive resolution of nested objects, when the step_call
        contains a dot (.) to access an object attribute.
        """
        method = None
        # If the step call is a class, get the default method of the class.
        if type(step_call) is type:
            # return getattr(step_call, self._default_object_method)
            return getattr(step_call, '__init__')

        if hasattr(self.host, step_call):
            method = getattr(self.host, step_call)
        elif hasattr(self, step_call):
            method = getattr(self, step_call)
        elif step_call in globals():
            method = globals()[step_call]
        # Check if 'step_call' contains a dot (.) and if so, try to get the
        # method from the object after the dot.
        elif '.' in step_call:
            obj_name, method_name = step_call.split('.')
            if hasattr(self.host, obj_name):
                obj = getattr(self.host, obj_name)
                method = getattr(obj, method_name)
            else:
                raise ValueError(
                    f"Object {obj_name} not found in host object")
        else:
            raise ValueError(
                f"Parameter {step_call} not found in host object or globals")
                
        return method

    def _get_params(self, method, param_names) -> List[Any]:
        """
        Get the parameters from the host object.

        Parameters
        ----------
        param_names: list
            List of parameter names to get from the host object.

        Returns
        -------
        params: list
            List of parameters from the host object.
        """
        params = []
        for arg in param_names:
            if type(arg) is str:
                if not hasattr(method, arg):
                    raise ValueError(
                        f"Parameter {arg} not found in host object or globals")
                param = getattr(method, arg)
            else:
                param = arg
            params.append(param)
        return params
    
    def _build_params(self, method_parameters, method_arguments) -> dict:
        params = {}
        for parameter, default_value in method_parameters.items():
            # If the parameter is in method_arguments, use the value from method_arguments.
            if parameter in method_arguments:
                # Two possibilities here: either the parameter is a normal value, 
                # in which case we simply take it, or is the name of an object created
                # in a previous step, in which case we take the object.
                if method_arguments[parameter] in self._objects.keys():
                    params[parameter] = self._objects[method_arguments[parameter]]
                else:
                    params[parameter] = method_arguments[parameter]
                continue
            # or if the parameter has a default value, use it.
            elif default_value is not inspect.Parameter.empty:
                params[parameter] = default_value
                continue
            # Otherwise, try to get the parameter from the host object or globals.
            if hasattr(self.host, parameter):
                params[parameter] = getattr(self.host, parameter)
            elif parameter in globals():
                params[parameter] = globals()[parameter]
            else:
                raise ValueError(
                    f"Parameter \'{parameter}\' not found in host object or globals")
        return params
    
    def run(self, steps: list, desc: str = "Running pipeline"):
        """
        Run the pipeline.

        Parameters
        ----------
        steps: dict
            Dictionary containing the steps to be run. Each key can be a tuple
            containing the name of the attribute to be created in the host object
            and the function or class to be called. But also, each key can be
            a function or method name. In the case of a tuple, the value returned by
            the function or the fit method of the class will be assigned to the
            attribute of the host object. In the case of a function or method name,
            the value returned by the function or the fit method of the class will
            not be assigned to any attribute of the host object.
            The value of each key is a list of parameters to be passed to the 
            function or class. Each parameter must be a string corresponding to
            an attribute of the host object or a value.
        """
        self._pbar = tqdm(total=len(steps), 
                          **tqdm_params(desc, self._prog_bar, leave=False, position=0))
        self._pbar.update(0)
        print("-"*80) if self._verbose else None

        for step_name in steps:
            print(f"Running step {step_name}") if self._verbose else None

            vble_name, step_call, step_parameters, step_arguments = \
                self._get_step_components(step_name)
            
            step_parameters = self._build_params(step_parameters, step_arguments)
            return_value = self.run_step(step_call, step_parameters)
            if vble_name is not None:
                setattr(self.host, vble_name, return_value)
                # Check if the new attribute created is an object and if so, 
                # add it to the list of objects.
                if type(return_value) is not type:
                    self._objects[vble_name] = return_value
                print(f"      New attribute {type(vble_name)}: \'{getattr(self.host, vble_name)}\'") \
                    if self._verbose else None
                
            print("-"*80) if self._verbose else None
            self._pbar_update(1)

        self._pbar.close()

    def run_step(self, step_name: Union[Any, str], list_of_params: List[Any] = []) -> Any:
        """
        Run a step of the pipeline.

        Parameters
        ----------
        step_name: str
            Function or class to be called.
        list_of_params: list
            List of parameters to be passed to the function or class.

        Returns
        -------
        return_value: any
            Value returned by the function or the fit method of the class.
        """
        return_value = None

        # Check if step_name is a function or a class already in globals
        if step_name in globals():
            step_name = globals()[step_name]
            # check if type of step_name is a function
            if type(step_name) is types.FunctionType or type(step_name) is types.MethodType:
                return_value = step_name(**list_of_params)
            # check if type of step_name is a class
            elif type(step_name) is type:
                obj = step_name(**list_of_params)
                obj.fit()
                return_value = obj
            else:
                raise TypeError("step_name must be a class or a function")
        # Check if step_name is a function or a class in the calling module
        elif not isinstance(step_name, str) and hasattr(step_name, '__module__'):
            # check if type of step_name is a function
            if type(step_name) is types.FunctionType or type(step_name) is types.MethodType:
                return_value = step_name(**list_of_params)
            # check if type of step_name is a class
            elif type(step_name) is type:
                obj = step_name(**list_of_params)
                return_value = obj
            else:
                raise TypeError("step_name must be a class or a function")
        # Check if step_name is a method of the host object
        elif hasattr(self.host, step_name):
            step_name = getattr(self.host, step_name)
            # check if type of step_name is a function
            if type(step_name) is types.FunctionType or type(step_name) is types.MethodType:
                return_value = step_name(**list_of_params)
            else:
                raise TypeError(
                    "step_name inside host object must be a function")
        # Consider that step_name is a method of some of the intermediate objects
        # in the pipeline
        else:
            # check if step name is of the form object.method
            if '.' not in step_name:
                raise ValueError(
                    f"step_name ({step_name}) must be method of an object: object.method")
            method_call = step_name
            root_object = self.host
            while '.' in method_call:
                call_composition = step_name.split('.')
                obj_name = call_composition[0]
                method_name = method_name = '.'.join(call_composition[1:])
                obj = getattr(root_object, obj_name)
                call_name = getattr(obj, method_name)
                method_call = '.'.join(method_name.split('.')[1:])
            return_value = call_name(**list_of_params)

        print("  > Return value:", type(return_value)) if self._verbose else None
        return return_value

    def _pbar_update(self, step=1):
        self._pbar.update(step)
        self._pbar.refresh()


if __name__ == "__main__":
    host = Host('host_value1', 'host_value2')
    pipeline = Pipeline(host, prog_bar=False, verbose=True)
    print(f"Host object: {host}")
    print(f"Pipeline initiated: {pipeline}")

    steps = [
        ('myobject1', SampleClass),
        ('method_with_object', {'obj': 'myobject1'})
        ('r1', 'm1'),
        ('r2', 'm2', {'what': 'new_what_value'}),
        ('myobject2', SampleClass, {'param2': True}),
        ('myobject2.fit'),
        'myobject2.method',
        'host_method',
        ('my_method')
    ]

    pipeline.run(steps)
