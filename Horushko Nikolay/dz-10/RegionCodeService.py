class RegionToken:

    def __init__(self, code: int, token: str) -> None:
        self.code = code
        self.token = token

    def isMatch(self, region_name: str) -> bool:
        return self.token in region_name


class RegionTokenService:

    def __init__(self, filename: str) -> None:
        self.region_tokens = self._read_region_codes(filename)

    def _read_region_codes(self, filename: str) -> list:
        lines = open(filename, 'r', encoding='UTF-8').readlines()
        codes = list()
        for line in lines:
            splitted = line.split(' ')
            codes.append(RegionToken(int(splitted[0]), splitted[1].upper().replace('\n', '')))
        return codes

    def map_code(self, region_name) -> int:
        if region_name != region_name:
            return 95
        for region_token in self.region_tokens:
            if region_token.isMatch(region_name):
                return region_token.code
        raise Exception(f"Code for region name: {region_name} was not found")
