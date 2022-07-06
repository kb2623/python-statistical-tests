# encoding=utf8
from __future__ import annotations
import math
from typing import Iterable, Dict, List, Tuple, Any, Optional, Callable, Union

import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker

class ProxyDelegate:
	r"""TODO.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		* proxy_type: TODO
		* fn_name: TODO
	"""
	def __init__(self, fn_name, proxy_type) -> None:
		r"""Construct TODO.

		Args:
			fn_name: TODO.
			proxy_type: TODO.
		"""
		self.proxy_type = proxy_type
		self.fn_name = fn_name

	def __get__(self, obj, objtype=None):
		r"""Get TODO.

		Args:
			obj: TODO.
			objtype: TODO.

		Returns:
			TODO
		"""
		return self.proxy_type(self.fn_name, obj)

class TaggedValueMeta(type):
	r"""TODO.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT
	"""
	def __init__(self, name, bases, dict) -> None:
		r"""Construct TODO.

		Args:
			name: TODO.
			bases: TODO.
			dict: TODO.
		"""
		for fn_name in self._proxies:
			try: getattr(self, fn_name)
			except AttributeError: setattr(self, fn_name, ProxyDelegate(fn_name, self._proxies[fn_name]))

class PassThroughProxy:
	r"""TODO.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		* fn_name: TODO.
		* obj: TODO
	"""
	def __init__(self, fn_name, obj) -> None:
		r"""Construct TODO.

		Args:
			fn_name: TODO.
			obj: TODO.
		"""
		self.fn_name = fn_name
		self.target = obj.proxy_target

	def __call__(self, *args):
		r"""Get TODO.

		Args:
			args: TODO.

		Returns:
			TODO
		"""
		fn = getattr(self.target, self.fn_name)
		ret = fn(*args)
		return ret

class ConvertArgsProxy(PassThroughProxy):
	r"""TODO.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		unit: TODO

	See Also:
		:class:`NiaPy.util.basc_unit.PassThroughProxy`
	"""
	def __init__(self, fn_name, obj) -> None:
		r"""Contruct TODO.

		Args:
			fn_name: TODO.
			obj: TODO.
		"""
		PassThroughProxy.__init__(self, fn_name, obj)
		self.unit = obj.unit

	def __call__(self, *args):
		r"""Get TODO.

		Args:
			args: TODO

		Returns:
			TODO.
		"""
		converted_args = []
		for a in args:
			try: converted_args.append(a.convert_to(self.unit))
			except AttributeError: converted_args.append(TaggedValue(a, self.unit))
		converted_args = tuple([c.get_value() for c in converted_args])
		return PassThroughProxy.__call__(self, *converted_args)

class ConvertReturnProxy(PassThroughProxy):
	r"""TODO.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		unit: TODO.

	See Also:
		:class:`NiaPy.util.basc_unit.PassThroughProxy`
	"""
	def __init__(self, fn_name, obj) -> None:
		r"""Construct TODO.

		Args:
			fn_name: TODO.
			obj: TODO.
		"""
		PassThroughProxy.__init__(self, fn_name, obj)
		self.unit = obj.unit

	def __call__(self, *args):
		r"""Get TODO.

		Args:
			args: TODO.

		Returns:
			TODO
		"""
		ret = PassThroughProxy.__call__(self, *args)
		return (NotImplemented if ret is NotImplemented else TaggedValue(ret, self.unit))

class ConvertAllProxy(PassThroughProxy):
	r"""TODO.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		* unit: TODO

	See Also:
		:class:`NiaPy.util.basc_unit.PassThroughProxy`
	"""
	def __init__(self, fn_name, obj) -> None:
		r"""Construct ConvertAllProxy object.

		Args:
			fn_name: TODO.
			obj: TODO.
		"""
		PassThroughProxy.__init__(self, fn_name, obj)
		self.unit = obj.unit

	def __call__(self, *args) -> TaggedValue:
		r"""Get the tagged values.

		If this arg has a unit type but no conversion ability, this operation is prohibited

		Args:
			args: TODO

		Returns:
			TODO
		"""
		converted_args = []
		arg_units = [self.unit]
		for a in args:
			if hasattr(a, 'get_unit') and not hasattr(a, 'convert_to'):
								return NotImplemented
			if hasattr(a, 'convert_to'):
				try: a = a.convert_to(self.unit)
				except Exception: pass
				arg_units.append(a.get_unit())
				converted_args.append(a.get_value())
			else:
				converted_args.append(a)
				if hasattr(a, 'get_unit'): arg_units.append(a.get_unit())
				else: arg_units.append(None)
		converted_args = tuple(converted_args)
		ret = PassThroughProxy.__call__(self, *converted_args)
		if ret is NotImplemented: return NotImplemented
		ret_unit = unit_resolver(self.fn_name, arg_units)
		if ret_unit is NotImplemented: return NotImplemented
		return TaggedValue(ret, ret_unit)

class TaggedValue(metaclass=TaggedValueMeta):
	r"""TODO.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		_proxies: TODO
		value: TODO
		unit: TODO
		proxy_target: TODO
	"""
	_proxies: Dict[str, ConvertAllProxy] = {
		'__add__': ConvertAllProxy,
		'__sub__': ConvertAllProxy,
		'__mul__': ConvertAllProxy,
		'__rmul__': ConvertAllProxy,
		'__cmp__': ConvertAllProxy,
		'__lt__': ConvertAllProxy,
		'__gt__': ConvertAllProxy,
		'__len__': PassThroughProxy
	}

	def __new__(cls, value, unit) -> TaggedValue:
		r"""Generate a new subclass for value.

		Args:
			value: TODO
			unit: TODO

		Returns:
			TODO
		"""
		value_class = type(value)
		try:
			subcls = type(f'TaggedValue_of_{value_class.__name__}', (cls, value_class), {})
			if subcls not in units.registry: units.registry[subcls] = basicConverter
			return object.__new__(subcls)
		except TypeError:
			if cls not in units.registry: units.registry[cls] = basicConverter
			return object.__new__(cls)

	def __init__(self, value: Union[int, float, Any], unit: BasicUnit) -> None:
		r"""Construct TODO.

		Args:
			value: TODO
			unit: TODO
		"""
		self.value = value
		self.unit = unit
		self.proxy_target = self.value

	def __getattribute__(self, name: str):
		r"""Get attribute.

		Args:
			name: TODO

		Returns:
			TODO
		"""
		if name.startswith('__'): return object.__getattribute__(self, name)
		variable = object.__getattribute__(self, 'value')
		if hasattr(variable, name) and name not in self.__class__.__dict__: return getattr(variable, name)
		return object.__getattribute__(self, name)

	def __array__(self, dtype: Any = object) -> np.ndarray:
		r"""Get array of TODO.

		Args:
			dtype: Type of return values.

		Returns:
			Array of values of dtype.
		"""
		return np.asarray(self.value).astype(dtype)

	def __array_wrap__(self, array, context) -> TaggedValue:
		r"""Get wraped array.

		Args:
			array: TODO
			context: TODO

		Returns:
			TODO
		"""
		return TaggedValue(array, self.unit)

	def __repr__(self) -> str:
		r"""Get string representation of this object.

		Returns:
			String formated representation object values.
		"""
		return 'TaggedValue({!r}, {!r})'.format(self.value, self.unit)

	def __str__(self) -> str:
		r"""Get string representation of values of object.

		Returns:
			String representation of object values.
		"""
		return str(self.value) + ' in ' + str(self.unit)

	def __len__(self) -> int:
		r"""Get number of values.

		Returns:
			Number of elements.
		"""
		return len(self.value)

	def __iter__(self) -> Iterable[TaggedValue]:
		r"""Get generator for values.

		Returns:
			Generator expression rather than use `yield`, so that

		Raises:
			TypeError is raised by iter(self) if appropriate when checking for iterability.
		"""
		return (TaggedValue(inner, self.unit) for inner in self.value)

	def get_compressed_copy(self, mask: Union[bool, List[bool], Tuple[bool], np.ndarray]) -> TaggedValue:
		r"""Get compressed copy of values.

		Args:
			mask: Mask that determins which values to use.

		Returns:
			TODO.
		"""
		new_value = np.ma.masked_array(self.value, mask=mask).compressed()
		return TaggedValue(new_value, self.unit)

	def convert_to(self, unit) -> TaggedValue:
		r"""Convert unit.

		Args:
			unit: TODO.

		Returns:
			TODO.
		"""
		if unit == self.unit or not unit: return self
		try: new_value = self.unit.convert_value_to(self.value, unit)
		except AttributeError: new_value = self
		return TaggedValue(new_value, unit)

	def get_value(self) -> Union[int, float, Any]:
		r"""Get value.

		Returns:
			TODO.
		"""
		return self.value

	def get_unit(self) -> BasicUnit:
		r"""Get unit.

		Returns:
			Unit.
		"""
		return self.unit

class BasicUnit:
	r"""BasicUnit class.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name: TODO.
		fullname: TODO.
		conversions: TODO
	"""
	def __init__(self, name: str, fullname: Optional[str] = None) -> None:
		r"""Construct BasicUnit.

		Args:
			name: TODO.
			fullname: TODO.
		"""
		self.name = name
		if fullname is None: fullname = name
		self.fullname = fullname
		self.conversions = dict()

	def __repr__(self) -> str:
		r"""Get string formated values.

		Returns:
			Formated string with values.
		"""
		return f'BasicUnit({self.name})'

	def __str__(self) -> str:
		r"""Get string values.

		Returns:
			String with values.
		"""
		return self.fullname

	def __call__(self, value) -> TaggedValue:
		r"""TODO.

		Args:
			value: TODO

		Returns:
			TODO.
		"""
		return TaggedValue(value, self)

	def __mul__(self, rhs: Union[int, float, Any]) -> TaggedValue:
		r"""Multiply unit with value.

		Args:
			rhs: Multiplication factor.

		Returns:
			Muliplied value.
		"""
		value, unit = rhs, self
		if hasattr(rhs, 'get_unit'):
			value, unit = rhs.get_value(), rhs.get_unit()
			unit = unit_resolver('__mul__', (self, unit))
		if unit is NotImplemented: return NotImplemented
		return TaggedValue(value, unit)

	def __rmul__(self, lhs: Union[int, float, Any]) -> BasicUnit:
		r"""Multiply unit with value.

		Args:
			lhs: TODO

		Returns:
			TODO.
		"""
		return self * lhs

	def __array_wrap__(self, array, context) -> TaggedValue:
		r"""TODO.

		Args:
			array: TODO.
			context: TODO.

		Returns:
			TODO.
		"""
		return TaggedValue(array, self)

	def __array__(self, t: Optional[type] = None, context: Optional[Any] = None) -> np.ndarray:
		r"""Get array.

		Args:
			t:
			context:

		Returns:

		"""
		ret = np.array([1])
		if t is not None: return ret.astype(t)
		else: return ret

	def add_conversion_factor(self, unit: BasicUnit, factor: Union[int, float]) -> None:
		r"""Add conversion factor.

		Args:
			unit:
			factor:

		Returns:

		"""
		def convert(x): return x * factor
		self.conversions[unit] = convert

	def add_conversion_fn(self, unit, fn) -> None:
		r"""Add conversion function.

		Args:
			unit:
			fn:

		Returns:

		"""
		self.conversions[unit] = fn

	def get_conversion_fn(self, unit: BasicUnit) -> Callable[[Union[float, Any]], Union[float, Any]]:
		r"""Get function for converting values.

		Args:
			unit:

		Returns:

		"""
		return self.conversions[unit]

	def convert_value_to(self, value: Union[int, float, Any], unit: BasicUnit) -> Union[float, Any]:
		r"""Convert value to unit type.

		Args:
			value:
			unit:

		Returns:

		"""
		conversion_fn = self.conversions[unit]
		ret = conversion_fn(value)
		return ret

	def get_unit(self) -> BasicUnit:
		r"""Get unit type.

		Returns:
			Unit.
		"""
		return self

class UnitResolver:
	r"""Class UnitResolver.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT

	Attributes:
		op_dict: Mapping from str to callable.
	"""
	def addition_rule(self, units: BasicUnit) -> TaggedValue:
		r"""Additional rule funciton.

		Args:
			units:

		Returns:

		"""
		for unit_1, unit_2 in zip(units[:-1], units[1:]):
			if unit_1 != unit_2: return NotImplemented
		return units[0]

	def multiplication_rule(self, units: BasicUnit) -> TaggedValue:
		r"""Multiplication rule funciton.

		Args:
			units:

		Returns:

		"""
		non_null = [u for u in units if u]
		if len(non_null) > 1: return NotImplemented
		return non_null[0]

	op_dict: Dict[str, Callable[[BasicUnit], TaggedValue]] = {
		'__mul__': multiplication_rule,
		'__rmul__': multiplication_rule,
		'__add__': addition_rule,
		'__radd__': addition_rule,
		'__sub__': addition_rule,
		'__rsub__': addition_rule
	}

	def __call__(self, operation: str, units: BasicUnit) -> Callable[[BasicUnit], TaggedValue]:
		r"""Get TODO.

		Args:
			operation:
			units:

		Returns:

		"""
		if operation not in self.op_dict: return NotImplemented
		return self.op_dict[operation](self, units)

unit_resolver = UnitResolver()

cm = BasicUnit('cm', 'centimeters')
inch = BasicUnit('inch', 'inches')
inch.add_conversion_factor(cm, 2.54)
cm.add_conversion_factor(inch, 1 / 2.54)

radians = BasicUnit('rad', 'radians')
degrees = BasicUnit('deg', 'degrees')
radians.add_conversion_factor(degrees, 180.0 / np.pi)
degrees.add_conversion_factor(radians, np.pi / 180.0)

secs = BasicUnit('s', 'seconds')
hertz = BasicUnit('Hz', 'Hertz')
minutes = BasicUnit('min', 'minutes')

secs.add_conversion_fn(hertz, lambda x: 1. / x)
secs.add_conversion_factor(minutes, 1 / 60.0)

# radians formatting
def rad_fn(x: Union[int, float], pos=None) -> str:
	r"""Get convert to radian string.

	Args:
		x:
		pos:

	Returns:

	"""
	if x >= 0: n = int((x / np.pi) * 2.0 + 0.25)
	else: n = int((x / np.pi) * 2.0 - 0.25)
	if n == 0: return '0'
	elif n == 1: return r'$\pi/2$'
	elif n == 2: return r'$\pi$'
	elif n == -1: return r'$-\pi/2$'
	elif n == -2: return r'$-\pi$'
	elif n % 2 == 0: return fr'${n//2}\pi$'
	else: return fr'${n}\pi/2$'

class BasicUnitConverter(units.ConversionInterface):
	r"""BasicUnitConverter class.

	Date:
		2019

	Author
		Klemen Berkovič

	License:
		MIT
	"""
	@staticmethod
	def axisinfo(unit: BasicUnit, axis: int) -> Optional[units.AxisInfo]:
		r"""Get axis info.

		Args:
			unit:
			axis:

		Returns:

		"""
		if unit == radians: return units.AxisInfo(majloc=ticker.MultipleLocator(base=np.pi / 2), majfmt=ticker.FuncFormatter(rad_fn), label=unit.fullname,)
		elif unit == degrees: return units.AxisInfo(majloc=ticker.AutoLocator(), majfmt=ticker.FormatStrFormatter(r'$%i^\circ$'), label=unit.fullname,)
		elif unit is not None:
			if hasattr(unit, 'fullname'): return units.AxisInfo(label=unit.fullname)
			elif hasattr(unit, 'unit'): return units.AxisInfo(label=unit.unit.fullname)
		return None

	@staticmethod
	def convert(val: Union[int, float, Any], unit: BasicUnit, axis: int) -> Union[int, float, Any]:
		r"""Convert value with respect to units.

		Args:
			val:
			unit:
			axis:

		Returns:

		"""
		if units.ConversionInterface.is_numlike(val): return val
		if np.iterable(val):
			if isinstance(val, np.ma.MaskedArray): val = val.astype(float).filled(np.nan)
			out = np.empty(len(val))
			for i, thisval in enumerate(val):
				if np.ma.is_masked(thisval): out[i] = np.nan
				else:
					try: out[i] = thisval.convert_to(unit).get_value()
					except AttributeError: out[i] = thisval
			return out
		if np.ma.is_masked(val): return np.nan
		else: return val.convert_to(unit).get_value()

	@staticmethod
	def default_units(x: Union[TaggedValue, Tuple[TaggedValue], List[TaggedValue], np.ndarray], axis: int) -> BasicUnit:
		r"""Get the default unit for x or None.

		Args:
			x: TODO
			axis: TODO

		Returns:
			Default unit for x.
		"""
		if np.iterable(x):
			for thisx in x: return thisx.unit
		return x.unit

def cos(x: TaggedValue) -> float:
	r"""Calculate cos values.

	Args:
		x:

	Returns:

	"""
	if np.iterable(x): return [math.cos(val.convert_to(radians).get_value()) for val in x]
	else: return math.cos(x.convert_to(radians).get_value())

basicConverter = BasicUnitConverter()
units.registry[BasicUnit] = basicConverter
units.registry[TaggedValue] = basicConverter

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
