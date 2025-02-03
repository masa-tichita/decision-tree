from pydantic import BaseModel


class Compas(BaseModel):
    age: str = "age"
    c_charge_degree: str = "c_charge_degree"
    race: str = "race"
    age_cat: str = "age_cat"
    score_text: str = "score_text"
    sex: str = "sex"
    priors_count: str = "priors_count"
    days_b_screening_arrest: str = "days_b_screening_arrest"
    decile_score: str = "decile_score"
    is_recid: str = "is_recid"
    two_year_recid: str = "two_year_recid"
    c_jail_in: str = "c_jail_in"
    c_jail_out: str = "c_jail_out"

    @classmethod
    def new(cls):
        return cls()


class Cols(BaseModel):
    compas: Compas = Compas.new()

    @classmethod
    def new(cls):
        return cls()


cols = Cols.new()

if __name__ == "__main__":
    cols = Cols.new()
    print(cols.compas.age)
