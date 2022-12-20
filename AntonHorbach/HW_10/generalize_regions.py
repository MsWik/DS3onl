def generalize_regions(src_region_name):
    region_name = str(src_region_name)
    if "МОСКОВ" in region_name or "МЫТИЩ" in region_name:
        return "МОСКОВСКАЯ ОБЛАСТЬ"
    elif "МОСКВ" in region_name:
        return "МОСКВА"
    elif "ПЕТЕРБ" in region_name or "98" in region_name:
        return "САНКТ-ПЕТЕРБУРГ"
    elif "АРХАН" in region_name:
        return "АРХАНГЕЛЬСКАЯ ОБЛАСТЬ"
    elif "ХАНТЫ" in region_name:
        return "ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНОЫЙ ОКРУГ"
    elif "КАРАЧ" in region_name:
        return "РЕСПУБЛИКА КАРАЧАЕВО-ЧЕРКЕСИЯ"
    elif "КАЛУЖ" in region_name:
        return "КАЛУЖСКАЯ ОБЛАСТЬ"
    elif "АДЫГ" in region_name:
        return "РЕСПУБЛИКА АДЫГЕЯ"
    elif "БАШКОРТ" in region_name:
        return "РЕСПУБЛИКА БАШКОРТОСТАН"
    elif "БУРЯТ" in region_name:
        return "РЕСПБЛИКА БУРЯТИЯ"
    elif "АЛТАЙСКИЙ" in region_name:
        return "АЛТАЙСКИЙ КРАЙ"
    elif "АЛТАЙ" in region_name and "РЕ" in region_name:
        return "РЕСПУБЛИКА АЛТАЙ"
    elif "ДАГЕС" in region_name:
        return "РЕСПБЛИКА ДАГЕСТАН"
    elif "ИНГУШ" in region_name:
        return "РЕСПБЛИКА ИНГУШЕТИЯ"
    elif "КАБАРДИН" in region_name:
        return "КАБАРИНО-БАЛКАРСКАЯ РЕСПБЛИКА"
    elif "КАЛМЫК" in region_name:
        return "РЕСПБЛИКА КАЛМЫКИЯ"
    elif "КАРЕЛ" in region_name:
        return "РЕСПБЛИКА КАРЕЛИЯ"
    elif "КОМИ" in region_name:
        return "РЕСПБЛИКА КОМИ"
    elif "БУРЯТ" in region_name:
        return "РЕСПБЛИКА БУРЯТИЯ"
    elif "МАРИЙ" in region_name:
        return "РЕСПБЛИКА МАРИЙ ЭЛ"
    elif "МОРДОВ" in region_name:
        return "РЕСПБЛИКА МОРДОВИЯ"
    elif "ЯКУТ" in region_name or "САХА" in region_name:
        return "РЕСПБЛИКА САХА (ЯКУТИЯ)"
    elif "ОСЕТ" in region_name:
        return "РЕСПБЛИКА СЕВЕРНАЯ ОСЕТИЯ - АЛАНИЯ"
    elif "ТАТАР" in region_name:
        return "РЕСПБЛИКА ТАТАРСТАН"
    elif "ТЫВА" in region_name:
        return "РЕСПБЛИКА ТЫВА"
    elif "УДМУРТСК" in region_name:
        return "УДМУРТСКАЯ РЕСПБЛИКА"
    elif "ХАКАС" in region_name:
        return "РЕСПБЛИКА ХАКАСИЯ"
    elif "ЧУВАШ" in region_name:
        return "ЧУВАШСКАЯ РЕСПБЛИКА"
    elif "КРАСНОДАР" in region_name:
        return "КРАСНОДАРСКИЙ КРАЙ"
    elif "КРАСНОЯР" in region_name or "ЭВЕН" in region_name:
        return "КРАСНОЯРСКИЙ КРАЙ"
    elif "ДАЛЬНЕВОСТ" in region_name or "ВОСТО" in region_name:
        return "ДАЛЬНЕВОСТОЧНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ"
    elif "СТАВРОП" in region_name:
        return "СТАВОПОЛЬСКИЙ КРАЙ"
    elif "ХАБАР" in region_name:
        return "ХАБАРОВСКИЙ КРАЙ"
    elif "АМУР" in region_name:
        return "АМУРСКАЯ ОБЛАСТЬ"
    elif "АСТРАХАН" in region_name:
        return "АСТРАХАНСКАЯ ОБЛАСТЬ"
    elif "БРЯНС" in region_name:
        return "БРЯНСКАЯ ОБЛАСТЬ"
    elif "БЕЛГОР" in region_name:
        return "БЕЛГОРДОСКАЯ ОБЛАСТЬ"
    elif "ВЛАДИМИР" in region_name or "ГУСЬ" in region_name:
        return "ВЛАДИМИРСКАЯ ОБЛАСТЬ"
    elif "ВОЛГОГР" in region_name:
        return "ВОЛГОГРАДСКАЯ ОБЛАСТЬ"
    elif "ВОЛОГОД" in region_name:
        return "ВОЛОГОДСКАЯ ОБЛАСТЬ"
    elif "ВОРОН" in region_name:
        return "ВОРОНЕЖСКАЯ ОБЛАСТЬ"
    elif "ИВАНО" in region_name:
        return "ИВАНОВСКАЯ ОБЛАСТЬ"
    elif "ИРКУТ" in region_name:
        return "ИРКУТСКАЯ ОБЛАСТЬ"
    elif "КАЛИНИ" in region_name:
        return "КАЛИНГРАДСКАЯ ОБЛАСТЬ"
    elif "КАЛУЖ" in region_name:
        return "КАЛУЖСКАЯ ОБЛАСТЬ"
    elif "КАМЧАТ" in region_name:
        return "КАМЧАТСКИЙ КРАЙ"
    elif "КЕМЕР" in region_name:
        return "КЕМЕРОВСКАЯ ОБЛАСТЬ"
    elif "КИРОВ" in region_name:
        return "КИРОВСКАЯ ОБЛАСТЬ"
    elif "КОСТРОМ" in region_name:
        return "КОСТРОМСКАЯ ОБЛАСТЬ"
    elif "КУРГАН" in region_name:
        return "КУРГАНСКАЯ ОБЛАСТЬ"
    elif "КУРСК" in region_name:
        return "КУРСКАЯ ОБЛАСТЬ"
    elif "ЛЕНИН" in region_name:
        return "ЛЕНИНГРАДСКАЯ ОБЛАСТЬ"
    elif "ЛИПЕЦ" in region_name:
        return "ЛИПЕЦКАЯ ОБЛАСТЬ"
    elif "МАГАД" in region_name:
        return "МАГАДАНСКАЯ ОБЛАСТЬ"
    elif "МУРМ" in region_name:
        return "МУРМАНСКАЯ ОБЛАСТЬ"
    elif "НИЖЕГ" in region_name or "ПРИВОЛЖ" in region_name:
        return "НИЖЕГОРОДСКАЯ ОБЛАСТЬ"
    elif "НОВГОР" in region_name or "ГОРЬК" in region_name:
        return "НОВГОРОДСКАЯ ОБЛАСТЬ"
    elif "НОВОСИБ" in region_name or "ЧИТИН" in region_name:
        return "НОВОСИБИРСКАЯ ОБЛАСТЬ"
    elif "ОМСК" in region_name:
        return "ОМСКАЯ ОБЛАСТЬ"
    elif "ОРЕНБ" in region_name:
        return "ОРЕНБУРГСКАЯ ОБЛАСТЬ"
    elif "ОРЛО" in region_name or "ОРЕЛ" in region_name or "ОРЁЛ" in region_name:
        return "АСТРАХАНСКАЯ ОБЛАСТЬ"
    elif "ПЕНЗ" in region_name:
        return "ПЕНЗЕНСКАЯ ОБЛАСТЬ"
    elif "ОРЕНБ" in region_name:
        return "ОРЕНБУРГСКАЯ ОБЛАСТЬ"
    elif "ПЕРМ" in region_name:
        return "ПЕРМСКИЙ КРАЙ"
    elif "ПСКОВ" in region_name:
        return "ПСКОВСКАЯ ОБЛАСТЬ"
    elif "РОСТ" in region_name:
        return "РОСТОВСКАЯ КРАЙ"
    elif "РЯЗАН" in region_name:
        return "РЯЗАНСКАЯ ОБЛАСТЬ"
    elif "САМАР" in region_name:
        return "САМАРСКАЯ ОБЛАСТЬ"
    elif "САРАТОВ" in region_name:
        return "САРАТОВСКАЯ ОБЛАСТЬ"
    elif "САХАЛ" in region_name:
        return "САРАТОВСКАЯ ОБЛАСТЬ"
    elif "СВЕРДЛ" in region_name:
        return "СВЕРДЛОВСКАЯ ОБЛАСТЬ"
    elif "СМОЛЕН" in region_name:
        return "СМОЛЕНСКАЯ ОБЛАСТЬ"
    elif "ТАМБОВ" in region_name:
        return "ТАМБОВСКАЯ ОБЛАСТЬ"
    elif "ТВЕР" in region_name:
        return "ТВЕРСКАЯ ОБЛАСТЬ"
    elif "ТОМСК" in region_name:
        return "ТОМСКАЯ ОБЛАСТЬ"
    elif "ТУЛ" in region_name:
        return "ТУЛЬСКАЯ ОБЛАСТЬ"
    elif "ТЮМ" in region_name:
        return "ТЮМЕНСКАЯ ОБЛАСТЬ"
    elif "УЛЬЯ" in region_name:
        return "УЛЬЯНОВСКАЯ ОБЛАСТЬ"
    elif "ЧЕЛЯБ" in region_name or "74" in region_name:
        return "ЧЕЛЯБИНСКАЯ ОБЛАСТЬ"
    elif "БАЙКАЛ" in region_name:
        return "ЗАБАЙКАЛЬСКИЙ КРАЙ"
    elif "ЯРОСЛ" in region_name:
        return "ЯРОСЛАВСКАЯ ОБЛАСТЬ"
    elif "ЕВРЕ" in region_name:
        return "ЕВРЕЙСКАЯ АВТОНОМНАЯ ОБЛАСТЬ"
    elif "НЕНЕЦ" in region_name:
        return "НЕНЕЦКИЙ АВТОНОМНЫЙ ОКРУГ"
    elif "ЧУКОТ" in region_name:
        return "ЧУКОТСКИЙ АВТОНОМНЫЙ ОКРУГ"
    elif "ЧЕЧЕН" in region_name:
        return "ЧЕЧЕНСКАЯ РЕСПУБЛИКА"
    elif "ПРИМОР" in region_name:
        return "ПРИМОРСКИЙ КРАЙ"
    else:
        return "РОССИЯ"
