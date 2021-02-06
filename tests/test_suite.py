import unittest

import test_autograd
import test_nn


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromModule(test_autograd))
    suite.addTests(unittest.TestLoader().loadTestsFromModule(test_nn))

    # with open('UnittestTextReport.txt', 'a') as f:
    # runner = unittest.TextTestRunner(stream=f, verbosity=2)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
