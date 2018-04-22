import os
import sys
import unittest
from mock import Mock, MagicMock, patch, sentinel, call

import findspark
findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

sc = SparkContext('local[*]')
spark = SparkSession(sc)