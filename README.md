# Jason Nano + Yolo를 이용한 실시간 객체 탐지 Custom Python Code

### Yolo를 기반으로 작성한 실시간 or 영상의 프레임을 추출하여 설정해둔 범위 내 객체 유무 탐지 후 MQTT를 이용한 Firebase Realtime Database 값 저장 or 수정

### Custom Object Dataset을 통한 정밀 감지 및 불필요한 감지 제거

![image](https://github.com/Artecrowd/yolov5/assets/127479677/8a289e27-d830-493b-9e2e-d88a04d76b63)

##### Custom Object

객체 존재 유무만 판별하면 되기에 불필요한 라벨링 이름 제거

![image](https://github.com/Artecrowd/yolov5/assets/127479677/3c66807a-17de-4fc4-bbd8-8b1f683e22a5)


### 좌표 탐지를 통한 범위 탐지

![image](https://github.com/Artecrowd/yolov5/assets/127479677/9a390887-3e5b-4958-bb1f-aaef9776f646)

### torch_no_gard()를 이용한 백그라운드 쓰레드보다 효율적인 동시 작업

![image](https://github.com/Artecrowd/yolov5/assets/127479677/244394ce-c019-4092-aafc-0f09ba256c58)

### 실 사용 예시

![image](https://github.com/Artecrowd/yolov5/assets/127479677/8bafdf10-de19-41e6-8072-b2e259147a30)
