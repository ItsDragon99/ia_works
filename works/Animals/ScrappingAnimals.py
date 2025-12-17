import requests, os
import threading

def download_animals_images(query, num=500, out_dir="inat", page_start=1, total_downloaded=0):
    os.makedirs(f"{out_dir}/{query}", exist_ok=True)
    page = page_start
    downloaded = total_downloaded
    while downloaded < num:
        url = f"https://api.inaturalist.org/v1/observations?taxon_name={query}&license=cc0,cc-by,cc-by-nc&photos=true&page={page}"
        for attempt in range(5):
            try:
                data = requests.get(url).json()
                break
            except requests.exceptions.RequestException as e:
                print(f"error downloading page {page} for {query}: {e} retrying ({attempt+1}/5)")
        else:
            print(f"failed to download page {page} for {query} after 5 attempts, skipping")
            time.sleep(5)
            page += 1
            continue
        print(f"downloading {query}: page {page}, downloaded {downloaded}/{num}")
        if not data["results"]:
            break

        for obs in data["results"]:
            for photo in obs["photos"]:
                img_url = photo["url"].replace("square", "large")
                try:
                    img = requests.get(img_url).content
                    img_path = f"{out_dir}/{query}/{query}_{downloaded}.jpg"
                    with open(img_path, "wb") as f:
                        f.write(img)
                    downloaded += 1
                    if downloaded >= num:
                        break
                except:
                    pass 
        page += 1
        
def total_images_in_directory(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(".jpg")])


# Last log: Downloading canis lupus familiaris: Page 259, Downloaded 9971/10000
dog_page = 260

# Last log: Downloading felis catus: Page 210, Downloaded 9994/10000
cat_page = 211

# Last log: Downloading chelonia: Page 187, Downloaded 9956/10000
chen_page = 188

# Last log: Downloading formicidae: Page 191, Downloaded 9975/10000
formi_page = 192

# Last log: Downloading coccinellidae: Page 181, Downloaded 9960/10000
cocci_page = 182

animals_dir = "../dataset/animals"
dog_dir = f"{animals_dir}/canis lupus familiaris"
cat_dir = f"{animals_dir}/felis catus"
chen_dir = f"{animals_dir}/chelonia"
formi_dir = f"{animals_dir}/formicidae"
cocci_dir = f"{animals_dir}/coccinellidae"
total_dog_images = total_images_in_directory(dog_dir)
total_dog_images = total_dog_images -1 if total_dog_images >0 else 0
total_cat_images = total_images_in_directory(cat_dir)
total_cat_images = total_cat_images -1 if total_cat_images >0 else 0
total_chen_images = total_images_in_directory(chen_dir)
total_chen_images = total_chen_images -1 if total_chen_images >0 else 0
total_formi_images = total_images_in_directory(formi_dir)
total_formi_images = total_formi_images -1 if total_formi_images >0 else 0
total_cocci_images = total_images_in_directory(cocci_dir)
total_cocci_images = total_cocci_images -1 if total_cocci_images >0 else 0

print(f"Current images\nDogs: {total_dog_images}, \nCats: {total_cat_images}, \nChelonia: {total_chen_images}, \nFormicidae: {total_formi_images}, \nCoccinellidae: {total_cocci_images}")
threads = []
threads.append(threading.Thread(
    target=download_animals_images, args=("canis lupus familiaris", 10000), 
    kwargs={"out_dir": animals_dir, "page_start": dog_page, "total_downloaded": total_dog_images}
    ))
threads.append(threading.Thread(
    target=download_animals_images, args=("felis catus", 10000), 
    kwargs={"out_dir": animals_dir, "page_start": cat_page, "total_downloaded": total_cat_images}
    ))
threads.append(threading.Thread(
    target=download_animals_images, args=("chelonia", 10000), 
    kwargs={"out_dir": animals_dir, "page_start": chen_page, "total_downloaded": total_chen_images}
    ))
threads.append(threading.Thread(
    target=download_animals_images, args=("formicidae", 10000), 
    kwargs={"out_dir": animals_dir, "page_start": formi_page, "total_downloaded": total_formi_images}
    ))
threads.append(threading.Thread(
    target=download_animals_images, args=("coccinellidae", 10000), 
    kwargs={"out_dir": animals_dir, "page_start": cocci_page, "total_downloaded": total_cocci_images}
    ))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()