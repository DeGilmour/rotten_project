class Config(object):
    """Base config, uses staging database server."""
    TESTING = False
    DB_SERVER = '192.168.1.56'

    @property
    def DATABASE_URI(self):  # Note: all caps
        return f"mysql://user@{self.DB_SERVER}/foo"

class DevelopmentConfig(Config):
    DB_SERVER = 'localhost'
    FLASK_ENV="development"

class TestingConfig(Config):
    DB_SERVER = 'localhost'
    DATABASE_URI = 'sqlite:///:memory:'