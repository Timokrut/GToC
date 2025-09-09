import os
import json
import time
import asyncio
import aiofiles


from ultralytics import YOLO
from ChromeParser import ChromeParser   

model = YOLO('model/AtomV1_0_0.pt')

async def main(vehicle_number):
    parser = ChromeParser()

    # Connecting to site and entering vehicle number
    await parser.connect('https://sicmt.ru/fgis-taksi?type=car')
    await parser.pasteNumber(vehicle_number)

    # Donwloading and solving captcha
    await parser.downloadCAPTCHA(location=parser.captcha_location, filename=parser.captcha_filename)

    res = model(f'{parser.captcha_location}/{parser.captcha_filename}.png')

    sorted_symbols = await process_capthca(res)
 
    # Trying to send captcha, if there nowhere to paste - just a skip
    await parser.sendCAPTCHAsolution(sorted_symbols)
   
    await asyncio.sleep(3)

    # Parsing data into json file
    name_file = f'{vehicle_number}.json'
    
    data = await parser.click_and_parse_table()  #await parser.parseTable()
    
    await save_json(data, name_file)

    #await parser.click_on_table()


    await asyncio.sleep(2)

    #parser._click_and_parse_table()
    # await parser.click_export_button()
    # await asyncio.sleep(1)
    # await parser.downloadCAPTCHA(location=parser.captcha_location, filename=parser.captcha_filename)
    
    # res = model(f'{parser.captcha_location}/{parser.captcha_filename}.png')

    # sorted_symbols = await process_capthca(res)
    # # Trying to send captcha, if there nowhere to paste - just a skip
    # await parser.sendCAPTCHAsolution(sorted_symbols) 
    
    await asyncio.sleep(2)

    print(f'downloaded pdf for {vehicle_number}')
    print(f'finished {vehicle_number}')
    await parser.close(vehicle_number)
    return name_file
    

async def process_capthca(res):
    predicted_objects = res[0].boxes.xyxy.cpu().numpy()
    predicted_classes = res[0].boxes.cls.cpu().numpy()
    predicted_confidences = res[0].boxes.conf.cpu().numpy()

    results = [
        (model.names[int(predicted_classes[i])], predicted_confidences[i], *map(int, predicted_objects[i]))
        for i in range(len(predicted_objects))
    ]

    sorted_results = sorted(results, key=lambda x: x[2])
    sorted_symbols = ''.join([item[0] for item in sorted_results])
    return sorted_symbols

async def save_json(data, name_file):
    async with aiofiles.open(name_file, "w", encoding='utf-8') as file:
        await file.write(json.dumps(data, ensure_ascii=False, indent=4))

async def test():
    start_time = time.perf_counter()
    
    await asyncio.gather(
        #main('Т912ХА799'),
        main('В167ВА01'),
        # main('Е672РУ18'), 
        # main('ВУ52899'),
        # main('В562СС799'),
        # main('Х927НХ31'),
        # main('1')
    )
    end_time = time.perf_counter()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

async def test2():
    # asyncio.run(test())
    UUID = 'c2e7b69a-c510-4886-b3e5-888910397f12'
    url = f'https://sicmt.ru/fgis-taksi/car?id={UUID}'
    model = YOLO('model/AtomV1_0_0.pt')
    parser = ChromeParser()

    await parser.connect(url)
    await asyncio.sleep(3)
    await parser.click_export_button()
    await asyncio.sleep(3)

    await parser.downloadCAPTCHA(location=parser.captcha_location, filename=parser.captcha_filename)

    res = model(f'{parser.captcha_location}/{parser.captcha_filename}.png')

    sorted_symbols = await process_capthca(res)
    # Trying to send captcha, if there nowhere to paste - just a skip
    await parser.sendCAPTCHAsolution(sorted_symbols)
    asyncio.sleep(2) 


    os.system(f'mv {parser.save_pdf_path} ./{UUID}.pdf')


if __name__ == '__main__':
    asyncio.run(test2())