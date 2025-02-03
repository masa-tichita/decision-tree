from pydantic import BaseModel


class Param(BaseModel):
    depth: int = 2
    delta: float = 0.01
    time_limit: int = 300

    @classmethod
    def new(cls, depth, delta, time_limit):
        return cls(depth=depth, delta=delta, time_limit=time_limit)


params = Param.new(depth=2, delta=0.01, time_limit=300)
