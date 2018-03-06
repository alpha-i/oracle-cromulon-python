import unittest
import pytest

from alphai_cromulon_oracle.oracle import CromulonOracle


@pytest.mark.skip()
class TestOracleIntegration(unittest.TestCase):
    def setUp(self):
        self.oracle_class = CromulonOracle

    def get_data(self):
        self.data = make_dict_from_dataframe(gym_df)

    def tearDown(self):
        pass

    def test_basic_oracle_integration(self):
        oracle = CromulonOracle(self.configuration)
        oracle.train(full_data_dict, EXECUTION_TIME)

        prediction = oracle.predict(data_dict, EXECUTION_TIME, number_of_iterations=1)
