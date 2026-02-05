FROM node:14.3-stretch as builder

WORKDIR /app
COPY frontend /app
RUN cd filosui/ && npm install && yarn build

FROM node:14.3-stretch as webapp

WORKDIR /app
COPY --from=builder /app/filosui/build ./build/
COPY wait-for-it.sh /app
RUN yarn global add serve
CMD ["serve",  "-s", "build"]