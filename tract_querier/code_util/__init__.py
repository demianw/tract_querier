class DocStringInheritor(type):

    '''A variation on
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    by Paul McGuire
    '''
    def __new__(meta, name, bases, clsdict):
        if not('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
                doc = mro_cls.__doc__
                if doc:
                    clsdict['__doc__'] = doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()
                                if hasattr(mro_cls, attr)):
                    doc = getattr(getattr(mro_cls, attr), '__doc__')
                    if doc:
                        attribute.__doc__ = doc
                        break
        return type.__new__(meta, name, bases, clsdict)

import unittest
import sys


class Test(unittest.TestCase):

    def test_null(self):
        class Foo(object):

            def frobnicate(self):
                pass

        class Bar(Foo):
            __metaclass__ = DocStringInheritor
        self.assertEqual(Bar.__doc__, object.__doc__)
        self.assertEqual(Bar().__doc__, object.__doc__)
        self.assertEqual(Bar.frobnicate.__doc__, None)

    def test_inherit_from_parent(self):
        class Foo(object):

            'Foo'

            def frobnicate(self):
                'Frobnicate this gonk.'
        class Bar(Foo):
            __metaclass__ = DocStringInheritor
        self.assertEqual(Foo.__doc__, 'Foo')
        self.assertEqual(Foo().__doc__, 'Foo')
        self.assertEqual(Bar.__doc__, 'Foo')
        self.assertEqual(Bar().__doc__, 'Foo')
        self.assertEqual(Bar.frobnicate.__doc__, 'Frobnicate this gonk.')

    def test_inherit_from_mro(self):
        class Foo(object):

            'Foo'

            def frobnicate(self):
                'Frobnicate this gonk.'
        class Bar(Foo):
            pass

        class Baz(Bar):
            __metaclass__ = DocStringInheritor
        self.assertEqual(Baz.__doc__, 'Foo')
        self.assertEqual(Baz().__doc__, 'Foo')
        self.assertEqual(Baz.frobnicate.__doc__, 'Frobnicate this gonk.')

    def test_inherit_metaclass_(self):
        class Foo(object):

            'Foo'

            def frobnicate(self):
                'Frobnicate this gonk.'
        class Bar(Foo):
            __metaclass__ = DocStringInheritor

        class Baz(Bar):
            pass
        self.assertEqual(Baz.__doc__, 'Foo')
        self.assertEqual(Baz().__doc__, 'Foo')
        self.assertEqual(Baz.frobnicate.__doc__, 'Frobnicate this gonk.')


if __name__ == '__main__':
    sys.argv.insert(1, '--verbose')
    unittest.main(argv=sys.argv)
